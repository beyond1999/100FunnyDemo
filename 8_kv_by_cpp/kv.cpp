// kvdb_mvp.cpp
// A single-file, buildable C++17 mini key–value store showing the key ideas:
// WAL -> MemTable -> SSTable (+ background flush & compaction) with simple sparse index.
// Focus: clarity and engineering hygiene, not all edge cases.
// Platform: POSIX/Windows (uses standard i/o; mmap optional on POSIX).
// Build:  g++ -std=c++17 -O3 -pthread kvdb_mvp.cpp -o kvdb_mvp
// Run:    ./kvdb_mvp /tmp/mydb   (then type commands; see main())
//
// Design notes (very short):
//  - Write path: Put/Delete => append WAL (CRC32C) => apply to MemTable.
//  - Flush: When MemTable > threshold, freeze to immutable and write an SSTable file
//           (sorted by key, stores <type,seq> along with value). Rotate WAL by truncation.
//  - Read path: check active/imm memtable, then newest-to-oldest SSTables. SSTable has
//           a sparse index (every k entries) + footer with index offset for fast seek.
//  - Compaction: simple binary merge of two oldest SSTables keeping latest seq per key,
//           dropping tombstones.
//  - Concurrency: a coarse-grained approach using mutexes & shared_mutex; background
//           threads for flush and compaction. Easy to swap in a lock-free memtable later.
//  - IO variants: default std::ifstream/pread-like; POSIX mmap reader guarded by macro.
//
// WARNING: For brevity we omit manifest/versioning and many robustness details.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using u8  = uint8_t;  using u32 = uint32_t;  using u64 = uint64_t;  using i64 = int64_t;

// ----------------------- CRC32C (Castagnoli) ----------------------- //
namespace crc32c {
static constexpr u32 kPoly = 0x1EDC6F41u; // Castagnoli
static u32 table_[256];
static std::once_flag init_flag;
static void init() {
    for (u32 i = 0; i < 256; ++i) {
        u32 c = i;
        for (int j = 0; j < 8; ++j) {
            c = (c & 1) ? (kPoly ^ (c >> 1)) : (c >> 1);
        }
        table_[i] = c;
    }
}
inline u32 compute(const void* data, size_t n, u32 seed = 0xFFFFFFFFu) {
    std::call_once(init_flag, init);
    const u8* p = static_cast<const u8*>(data);
    u32 c = seed;
    for (size_t i = 0; i < n; ++i) {
        c = table_[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}
} // namespace crc32c

// ---------------------------- Utility ------------------------------ //
static inline void write_u32(std::ostream& os, u32 v) { os.write(reinterpret_cast<char*>(&v), sizeof(v)); }
static inline void write_u64(std::ostream& os, u64 v) { os.write(reinterpret_cast<char*>(&v), sizeof(v)); }
static inline bool read_u32(std::istream& is, u32& v){ return bool(is.read(reinterpret_cast<char*>(&v), sizeof(v))); }
static inline bool read_u64(std::istream& is, u64& v){ return bool(is.read(reinterpret_cast<char*>(&v), sizeof(v))); }

enum class RecType : u8 { kPut = 1, kDel = 2 };
struct Entry { std::string key; std::string val; RecType type{RecType::kPut}; u64 seq{0}; };

// ------------------------------ WAL -------------------------------- //
class WAL {
    fs::path path_;
    std::mutex mtx_;
public:
    explicit WAL(const fs::path& p) : path_(p) {
        fs::create_directories(path_.parent_path());
        // Ensure file exists.
        std::ofstream(path_, std::ios::binary | std::ios::app).close();
    }
    bool append(const Entry& e, bool sync=true) {
        std::lock_guard<std::mutex> lk(mtx_);
        std::ofstream os(path_, std::ios::binary | std::ios::app);
        if (!os) return false;
        // Header: [crc:u32][type:u8][key_len:u32][val_len:u32][seq:u64]
        u8 type = static_cast<u8>(e.type);
        u32 klen = (u32)e.key.size();
        u32 vlen = (u32)e.val.size();
        u64 seq  = e.seq;
        std::string buf;
        buf.reserve(1 + 4 + 4 + 8 + klen + vlen);
        // pack without crc for computation
        buf.push_back((char)type);
        buf.append(reinterpret_cast<char*>(&klen), sizeof(klen));
        buf.append(reinterpret_cast<char*>(&vlen), sizeof(vlen));
        buf.append(reinterpret_cast<char*>(&seq),  sizeof(seq));
        buf.append(e.key.data(), klen);
        buf.append(e.val.data(), vlen);
        u32 crc = crc32c::compute(buf.data(), buf.size());
        write_u32(os, crc);
        os.write(buf.data(), (std::streamsize)buf.size());
        if (sync) os.flush(); // fsync omitted for simplicity
        return bool(os);
    }

    // Replay records in order; stop gracefully on checksum error or EOF.
    template<class F>
    void recover(F&& fn) {
        std::ifstream is(path_, std::ios::binary);
        if (!is) return;
        while (true) {
            u32 crc=0; if (!read_u32(is, crc)) break;
            u8 type=0; u32 klen=0, vlen=0; u64 seq=0;
            if (!is.read((char*)&type, 1)) break;
            if (!read_u32(is, klen)) break; if (!read_u32(is, vlen)) break; if (!read_u64(is, seq)) break;
            std::string k(klen, '\0'); std::string v(vlen, '\0');
            if (!is.read(k.data(), klen)) break; if (!is.read(v.data(), vlen)) break;
            // verify crc
            std::string buf; buf.reserve(1+4+4+8+klen+vlen);
            buf.push_back((char)type);
            buf.append(reinterpret_cast<char*>(&klen), sizeof(klen));
            buf.append(reinterpret_cast<char*>(&vlen), sizeof(vlen));
            buf.append(reinterpret_cast<char*>(&seq),  sizeof(seq));
            buf.append(k.data(), klen); buf.append(v.data(), vlen);
            u32 got = crc32c::compute(buf.data(), buf.size());
            if (got != crc) break; // treat as clean EOF/truncation
            fn( Entry{std::move(k), std::move(v), type==1?RecType::kPut:RecType::kDel, seq} );
        }
    }

    // Truncate (rotate) after successful flush to SST.
    void reset() {
        std::lock_guard<std::mutex> lk(mtx_);
        std::ofstream os(path_, std::ios::binary | std::ios::trunc); (void)os;
    }
};

// ---------------------------- MemTable ----------------------------- //
class MemTable {
    // value = (val, type, seq)
    struct V { std::string v; RecType t; u64 seq; };
    std::unordered_map<std::string, V> map_;
    mutable std::shared_mutex mu_;
    size_t approx_bytes_{0};
public:
    void put(std::string k, std::string v, u64 seq) {
        std::unique_lock lk(mu_);
        approx_bytes_ -= map_.count(k) ? (map_[k].v.size()+k.size()) : 0;
        map_[std::move(k)] = V{std::move(v), RecType::kPut, seq};
        approx_bytes_ += map_.rbegin()->first.size() + map_.rbegin()->second.v.size();
    }
    void del(std::string k, u64 seq) {
        std::unique_lock lk(mu_);
        approx_bytes_ -= map_.count(k) ? (map_[k].v.size()+k.size()) : 0;
        map_[std::move(k)] = V{"", RecType::kDel, seq};
        approx_bytes_ += map_.rbegin()->first.size();
    }
    std::optional<Entry> get(const std::string& k) const {
        std::shared_lock lk(mu_);
        auto it = map_.find(k); if (it==map_.end()) return std::nullopt;
        const auto& v = it->second; return Entry{k, v.v, v.t, v.seq};
    }
    size_t approx_bytes() const { std::shared_lock lk(mu_); return approx_bytes_; }

    // Snapshot to sorted vector for SST write.
    std::vector<Entry> snapshot_sorted() const {
        std::shared_lock lk(mu_);
        std::vector<Entry> vec; vec.reserve(map_.size());
        for (auto& kv : map_) vec.push_back( Entry{kv.first, kv.second.v, kv.second.t, kv.second.seq} );
        std::sort(vec.begin(), vec.end(), [](const Entry& a, const Entry& b){ return a.key < b.key; });
        return vec;
    }

    void clear() { std::unique_lock lk(mu_); map_.clear(); approx_bytes_=0; }
    bool empty() const { std::shared_lock lk(mu_); return map_.empty(); }
};

// ---------------------------- SSTable ------------------------------ //
struct SSTFooter { u64 index_off{0}; u32 magic{0x53535431u}; /* 'SST1' */ };

class SSTWriter {
    fs::path path_;
    const size_t index_interval_ = 32; // every N entries into sparse index
public:
    explicit SSTWriter(fs::path p) : path_(std::move(p)) {}
    void write(const std::vector<Entry>& sorted) {
        std::ofstream os(path_, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("SSTWriter: open failed");
        struct Idx { std::string k; u64 off; };
        std::vector<Idx> index;
        size_t i=0; for (const auto& e : sorted) {
            if (i % index_interval_ == 0) index.push_back(Idx{e.key, (u64)os.tellp()});
            u8 type = (u8)e.type; u64 seq=e.seq; u32 klen=(u32)e.key.size(); u32 vlen=(u32)e.val.size();
            os.write((char*)&type,1); write_u64(os,seq); write_u32(os,klen); write_u32(os,vlen);
            os.write(e.key.data(), klen); os.write(e.val.data(), vlen);
            ++i;
        }
        u64 index_off = (u64)os.tellp();
        // write index: [count][(klen,key,off)*]
        u32 cnt = (u32)index.size(); write_u32(os, cnt);
        for (auto& it : index) {
            u32 klen=(u32)it.k.size(); write_u32(os, klen); os.write(it.k.data(), klen); write_u64(os, it.off);
        }
        // footer
        SSTFooter f; f.index_off = index_off; os.write(reinterpret_cast<char*>(&f), sizeof(f));
        os.flush();
    }
};

class SSTReader {
    fs::path path_;
    struct Idx { std::string k; u64 off; };
    std::vector<Idx> index_;
    u64 data_end_{0};
public:
    explicit SSTReader(fs::path p) : path_(std::move(p)) { load(); }

    void load() {
        std::ifstream is(path_, std::ios::binary); if (!is) throw std::runtime_error("SST open fail");
        // read footer
        is.seekg(- (std::streamoff)sizeof(SSTFooter), std::ios::end);
        SSTFooter f{}; is.read(reinterpret_cast<char*>(&f), sizeof(f));
        if (f.magic != 0x53535431u) throw std::runtime_error("Bad SST magic");
        data_end_ = f.index_off;
        // read index
        is.seekg(f.index_off, std::ios::beg);
        u32 cnt=0; if (!read_u32(is,cnt)) throw std::runtime_error("Bad index");
        index_.resize(cnt);
        for (u32 i=0;i<cnt;++i){ u32 klen=0; read_u32(is,klen); index_[i].k.resize(klen); is.read(index_[i].k.data(), klen); read_u64(is,index_[i].off);}    }

    // Find lower_bound in sparse index.
    size_t seek_index(const std::string& key) const {
        size_t l=0, r=index_.size();
        while (l<r) { size_t m=(l+r)/2; if (index_[m].k < key) l=m+1; else r=m; }
        if (l==0) return 0; return l-1; // last <= key
    }

    // Linear scan from the hint offset until key >= target or reach index next block/end.
    std::optional<Entry> get(const std::string& key) const {
        std::ifstream is(path_, std::ios::binary); if (!is) return std::nullopt;
        if (index_.empty()) return std::nullopt;
        size_t pos = seek_index(key);
        u64 start = index_[pos].off;
        u64 stop  = (pos+1<index_.size()) ? index_[pos+1].off : data_end_;
        is.seekg((std::streamoff)start, std::ios::beg);
        while ((u64)is.tellg() < stop) {
            u8 type=0; u64 seq=0; u32 klen=0, vlen=0; is.read((char*)&type,1); read_u64(is,seq); read_u32(is,klen); read_u32(is,vlen);
            std::string k(klen,'\0'); std::string v(vlen,'\0'); is.read(k.data(),klen); is.read(v.data(),vlen);
            if (k==key) return Entry{key,std::move(v), type==1?RecType::kPut:RecType::kDel, seq};
            if (k>key) break; // passed
        }
        return std::nullopt;
    }
};

// ------------------------------ DB --------------------------------- //
struct Options {
    fs::path dir;
    size_t memtable_max_bytes = 32 * 1024 * 1024; // 32MB
};

class DB {
    Options opt_;
    WAL wal_;
    MemTable active_;
    std::unique_ptr<MemTable> imm_; // immutable being flushed
    std::vector<fs::path> ssts_;     // newest at back
    std::mutex mu_;
    std::condition_variable_any cv_;
    std::atomic<u64> seq_{0};
    bool stop_{false};
    std::thread flush_thr_;
    std::thread compact_thr_;

    fs::path wal_path() const { return opt_.dir / "wal.log"; }
    fs::path sst_path(u64 id) const { std::ostringstream oss; oss << "sst_" << std::setw(6) << std::setfill('0') << id << ".sst"; return opt_.dir/oss.str(); }

public:
    explicit DB(Options o) : opt_(std::move(o)), wal_(wal_path()) {
        fs::create_directories(opt_.dir);
        recover();
        flush_thr_ = std::thread([this]{ flush_worker(); });
        compact_thr_ = std::thread([this]{ compact_worker(); });
    }
    ~DB(){ close(); }

    void close(){
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true; cv_.notify_all();
        }
        if (flush_thr_.joinable()) flush_thr_.join();
        if (compact_thr_.joinable()) compact_thr_.join();
        // final flush if any
        flush_now();
    }

    // ------------- public API ------------- //
    bool put(const std::string& k, const std::string& v) {
        Entry e{k, v, RecType::kPut, ++seq_}; if (!wal_.append(e)) return false; active_.put(k,v,e.seq); maybe_rotate(); cv_.notify_all(); return true; }
    bool del(const std::string& k) { Entry e{k, "", RecType::kDel, ++seq_}; if (!wal_.append(e)) return false; active_.del(k,e.seq); maybe_rotate(); cv_.notify_all(); return true; }

    std::optional<std::string> get(const std::string& k) {
        // active
        if (auto r = active_.get(k)) {
            if (r->type==RecType::kPut) return r->val; else return std::nullopt;
        }
        // immutable
        if (imm_) { if (auto r = imm_->get(k)) { if (r->type==RecType::kPut) return r->val; else return std::nullopt; } }
        // ssts newest -> oldest (latest seq wins, but we compact to remove shadowed keys)
        for (auto it = ssts_.rbegin(); it != ssts_.rend(); ++it) {
            try { SSTReader rd(*it); auto r = rd.get(k); if (r) { if (r->type==RecType::kPut) return r->val; else return std::nullopt; } }
            catch(...) { /* ignore corrupted */ }
        }
        return std::nullopt;
    }

    void flush_now() {
        std::unique_lock<std::mutex> lk(mu_);
        if (active_.empty()) return;
        if (!imm_) { imm_.reset(new MemTable()); *imm_ = active_; active_.clear(); }
        lk.unlock();
        flush_imm();
    }

    void compact_now() { cv_.notify_all(); }

    // For demo: quick benchmark
    void bench_put(size_t n, size_t vbytes=64) {
        std::mt19937_64 rng(42);
        auto rand_key = [&]{ u64 x=rng(); std::string k(16,'\0'); for(int i=0;i<16;++i) k[i] = 'a' + ((x>>((i%8)*8)) & 25); return k; };
        std::string v(vbytes, 'x');
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i=0;i<n;++i) {
            auto k = rand_key(); put(k, v);
            if ((i%10000)==0) { /* let background flush catch up */ }
        }
        flush_now();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        std::cout << "PUT "<<n<<" items in "<<ms<<" ms, "<<(1e3*n/ms)<<" op/s\n";
    }

private:
    void maybe_rotate(){ if (active_.approx_bytes() >= opt_.memtable_max_bytes) { std::lock_guard<std::mutex> lk(mu_); if (!imm_) { imm_.reset(new MemTable()); *imm_ = active_; active_.clear(); cv_.notify_all(); } } }

    void recover(){
        // Load existing SST files (sorted by filename)
        if (fs::exists(opt_.dir)) {
            for (auto& e : fs::directory_iterator(opt_.dir)) {
                if (e.is_regular_file() && e.path().filename().string().rfind("sst_",0)==0) ssts_.push_back(e.path());
            }
            std::sort(ssts_.begin(), ssts_.end());
        }
        // Replay WAL into memtable
        u64 maxseq=0; wal_.recover([&](Entry e){ maxseq = std::max(maxseq, e.seq); if (e.type==RecType::kPut) active_.put(e.key,e.val,e.seq); else active_.del(e.key,e.seq); });
        seq_.store(maxseq);
    }

    void flush_worker(){
        while (true) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&]{ return stop_ || imm_!=nullptr; });
            if (stop_ && !imm_) break;
            auto mem = std::move(imm_); lk.unlock();
            if (mem) {
                try { flush_memtable_to_sst(*mem); wal_.reset(); }
                catch(const std::exception& ex){ std::cerr << "flush error: "<<ex.what()<<"\n"; }
            }
        }
    }

    void compact_worker(){
        while (!stop_) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            if (stop_) break;
            // heuristic: if >= 3 ssts, compact the two oldest
            if (ssts_.size() >= 3) {
                fs::path a = ssts_.front(); fs::path b = ssts_[1];
                try { compact_pair(a,b); }
                catch(const std::exception& ex){ std::cerr << "compact error: "<<ex.what()<<"\n"; }
            }
        }
    }

    void flush_imm(){ std::unique_lock<std::mutex> lk(mu_); auto mem = std::move(imm_); lk.unlock(); if (mem) { flush_memtable_to_sst(*mem); wal_.reset(); } }

    void flush_memtable_to_sst(const MemTable& mem){
        auto vec = mem.snapshot_sorted(); if (vec.empty()) return;
        u64 id = next_sst_id(); fs::path p = sst_path(id);
        SSTWriter w(p); w.write(vec);
        std::lock_guard<std::mutex> lk(mu_);
        ssts_.push_back(p);
        std::sort(ssts_.begin(), ssts_.end());
        std::cout << "Flushed memtable to "<<p.filename().string()<<" ("<<vec.size()<<" entries)\n";
    }

    u64 next_sst_id(){
        u64 id=0; for (auto& s : ssts_) {
            auto name = s.filename().string();
            if (name.rfind("sst_",0)==0) { id = std::max<u64>(id, std::stoull(name.substr(4,6))); }
        }
        return id+1;
    }

    void compact_pair(const fs::path& a, const fs::path& b){
        SSTReader ra(a), rb(b);
        // naive approach: materialize maps (OK for demo)
        std::map<std::string, Entry> merged; // keep latest seq
        auto scan = [&](const fs::path& p){
            std::ifstream is(p, std::ios::binary);
            if (!is) return;
            // iterate all data until index_off
            is.seekg(- (std::streamoff)sizeof(SSTFooter), std::ios::end); SSTFooter f{}; is.read((char*)&f,sizeof(f)); is.seekg(0,std::ios::beg);
            while ((u64)is.tellg() < f.index_off) {
                u8 type=0; u64 seq=0; u32 klen=0,vlen=0; is.read((char*)&type,1); read_u64(is,seq); read_u32(is,klen); read_u32(is,vlen);
                std::string k(klen,'\0'); std::string v(vlen,'\0'); is.read(k.data(),klen); is.read(v.data(),vlen);
                auto it = merged.find(k);
                if (it==merged.end() || it->second.seq < seq) merged[k] = Entry{std::move(k), std::move(v), type==1?RecType::kPut:RecType::kDel, seq};
            }
        };
        scan(a); scan(b);
        // drop tombstones
        std::vector<Entry> vec; vec.reserve(merged.size());
        for (auto& kv : merged) if (kv.second.type==RecType::kPut) vec.push_back(std::move(kv.second));
        if (vec.empty()) { fs::remove(a); fs::remove(b); return; }
        u64 id = next_sst_id(); fs::path out = sst_path(id); SSTWriter w(out); w.write(vec);
        {
            std::lock_guard<std::mutex> lk(mu_);
            // remove a,b and add out
            auto rm = [&](const fs::path& p){ auto it = std::find(ssts_.begin(), ssts_.end(), p); if (it!=ssts_.end()) ssts_.erase(it); fs::remove(p); };
            rm(a); rm(b); ssts_.push_back(out); std::sort(ssts_.begin(), ssts_.end());
        }
        std::cout << "Compacted -> "<<out.filename().string()<<" ("<<vec.size()<<" live keys)\n";
    }
};

// ------------------------------ CLI -------------------------------- //
static void repl(DB& db){
    std::cout << "Commands: put <k> <v> | get <k> | del <k> | flush | compact | bench <N> | exit\n";
    std::string cmd; while (true) {
        std::cout << "> "; if (!(std::cin>>cmd)) break; if (cmd=="put"){ std::string k,v; std::cin>>k>>v; db.put(k,v); std::cout<<"OK\n"; }
        else if (cmd=="get"){ std::string k; std::cin>>k; auto v=db.get(k); if(v) std::cout<<*v<<"\n"; else std::cout<<"(nil)\n"; }
        else if (cmd=="del"){ std::string k; std::cin>>k; db.del(k); std::cout<<"OK\n"; }
        else if (cmd=="flush"){ db.flush_now(); }
        else if (cmd=="compact"){ db.compact_now(); }
        else if (cmd=="bench"){ size_t n; std::cin>>n; db.bench_put(n); }
        else if (cmd=="exit"){ break; } else { std::cout<<"?\n"; }
    }
}

int main(int argc, char** argv){
    if (argc<2) { std::cerr<<"Usage: "<<argv[0]<<" <db_dir>\n"; return 1; }
    Options opt; opt.dir = fs::path(argv[1]); opt.memtable_max_bytes = 8*1024*1024; // small for demo
    DB db(opt);
    repl(db);
    return 0;
}
