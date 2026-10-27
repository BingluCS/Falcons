#ifndef Falcons_HUFFMAN_ENCODER_HPP
#define Falcons_HUFFMAN_ENCODER_HPP

#include <cstdint>

#include "Falcons/def.hpp"
#include "Falcons/encoder/Encoder.hpp"
#include "Falcons/utils/ByteUtil.hpp"
#include "Falcons/utils/MemoryUtil.hpp"
#include "Falcons/utils/Timer.hpp"
#if INTPTR_MAX == INT64_MAX  // 64bit system
#include "Falcons/utils/ska_hash/unordered_map.hpp"
#endif  // INTPTR_MAX == INT64_MAX
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>


//#ifdef _OPENMP
//#include <omp.h>
//#endif


namespace Falcons {

template <class T>
class HuffmanEncoder : public concepts::EncoderInterface<T> {
   public:
    typedef struct node_t {
        struct node_t *left, *right;
        size_t freq;
        char t;  // in_node:0; otherwise:1
        T c;
    } *node;

    typedef struct HuffmanTree {
        unsigned int stateNum;
        unsigned int allNodes;
        struct node_t *pool;
        node *qqq, *qq;  // the root node of the HuffmanTree is qq[1]
        int n_nodes;     // n_nodes is for compression
        int qend;
        uint64_t *code;
        unsigned char *cout;
        int n_inode;  // n_inode is for decompression
        int maxBitCount;
    } HuffmanTree;

    HuffmanEncoder() {
        int x = 1;
        char *y = reinterpret_cast<char *>(&x);
        if (*y == 1)
            sysEndianType = 0;
        else  //=0
            sysEndianType = 1;
    }

    struct alignas(8) TableEntry {
        int symbol = 0;       
        int  code_len = 0;
        uint8_t  is_leaf = 0; 
    };

    int usingTable = 0;
    std::vector<TableEntry>              L1_table;
    int L1 = 10;   

    ~HuffmanEncoder() override { Falcons_FreeHuffman(); }

    // build huffman tree
    HuffmanTree *createHuffmanTree(int stateNum) {
        HuffmanTree *huffmanTree = static_cast<HuffmanTree *>(malloc(sizeof(HuffmanTree)));
        memset(huffmanTree, 0, sizeof(HuffmanTree));
        huffmanTree->stateNum = stateNum;
        huffmanTree->allNodes = 2 * stateNum;

        huffmanTree->pool = static_cast<struct node_t *>(malloc(huffmanTree->allNodes * 2 * sizeof(struct node_t)));
        huffmanTree->qqq = static_cast<node *>(malloc(huffmanTree->allNodes * 2 * sizeof(node)));
        huffmanTree->code = static_cast<uint64_t *>(malloc(huffmanTree->stateNum * sizeof(uint64_t)));
        huffmanTree->cout = static_cast<unsigned char *>(malloc(huffmanTree->stateNum * sizeof(unsigned char)));



        memset(huffmanTree->pool, 0, huffmanTree->allNodes * 2 * sizeof(struct node_t));
        memset(huffmanTree->qqq, 0, huffmanTree->allNodes * 2 * sizeof(node));
        memset(huffmanTree->code, 0, huffmanTree->stateNum * sizeof(uint64_t));
        memset(huffmanTree->cout, 0, huffmanTree->stateNum * sizeof(unsigned char));
        huffmanTree->qq = huffmanTree->qqq - 1;
        huffmanTree->n_nodes = 0;
        huffmanTree->n_inode = 0;
        huffmanTree->qend = 1;

        return huffmanTree;
    }

    /**
     * build huffman tree using bins
     * @param bins
     * @param stateNum
     */
    void preprocess_encode(const std::vector<T> &bins, int stateNum) override {
        preprocess_encode(bins.data(), bins.size(), stateNum);
    }

    /**
     * build huffman tree using bins
     * @param bins
     * @param num_bin
     * @param stateNum
     */
    void preprocess_encode(const T *bins, size_t num_bin, int stateNum) {
        nodeCount = 0;
        if (num_bin == 0) {
            throw std::invalid_argument("Huffman bins should not be empty");
        }
        init(bins, num_bin);
        for (unsigned int i = 0; i < huffmanTree->stateNum; i++)
            if (huffmanTree->cout[i] != 0) nodeCount++;
        nodeCount = nodeCount * 2 - 1;
    }

    // void preprocess_encode(const T *bins, size_t num_bin, int stateNum, size_t* frequencyList) {
    //     nodeCount = 0;
    //     if (num_bin == 0) {
    //         throw std::invalid_argument("Huffman bins should not be empty");
    //     }
    //     init(bins, num_bin, frequencyList);
    //     for (unsigned int i = 0; i < huffmanTree->stateNum; i++)
    //         if (huffmanTree->cout[i] != 0) nodeCount++;
    //     nodeCount = nodeCount * 2 - 1;
    // }
    /*
    // save the huffman Tree in the compressed data
    void save(uchar *&c) override {
        // auto cc = c;
        write(offset, c);
        int32ToBytes_bigEndian(c, nodeCount);
        c += sizeof(int);
        int32ToBytes_bigEndian(c, huffmanTree->stateNum / 2);
        c += sizeof(int);
        uint totalSize = 0;  // = convert_HuffTree_to_bytes_anyStates(nodeCount, c);
        // std::cout << "nodeCount = " << nodeCount << std::endl;
        if (nodeCount <= 256)
            totalSize = convert_HuffTree_to_bytes_anyStates<unsigned char>(nodeCount, c);
        else if (nodeCount <= 65536)
            totalSize = convert_HuffTree_to_bytes_anyStates<unsigned short>(nodeCount, c);
        else
            totalSize = convert_HuffTree_to_bytes_anyStates<unsigned int>(nodeCount, c);
        c += totalSize;
        //            return c - cc;
    }*/

    void save(uchar *&c) override {
        // 写 offset
        int x =(usingTable & 1u) << 31 | offset;
        write(x, c);
        // usingTable = 1;
        // write(usingTable, c);
        int stateNum = static_cast<int>(huffmanTree->stateNum);
        int32ToBytes_bigEndian(c, stateNum);
        c += sizeof(int);
        // 写最大码长（可选，但有用）
        //unsigned char maxLen = static_cast<unsigned char>(canonMaxLen);
        //*c++ = maxLen;

        // 写每个 state 的码长（0 表示该 state 未使用）
        for (int i = 0; i < stateNum; ++i) {
            *c++ = huffmanTree->cout[i];
        }
    }

    /*
    size_t size_est() override {
        size_t b = (nodeCount <= 256) ? sizeof(unsigned char)
                                      : ((nodeCount <= 65536) ? sizeof(unsigned short) : sizeof(unsigned int));
        return 1 + 2 * nodeCount * b + nodeCount * sizeof(unsigned char) + nodeCount * sizeof(T) + sizeof(int) +
               sizeof(int) + sizeof(T);
    }*/
    size_t size_est() override {
        // offset + stateNum(int) + maxLen(1 byte) + stateNum bytes of length
        return sizeof(T) + sizeof(int) + 1 + huffmanTree->stateNum * sizeof(unsigned char);
    }    

    size_t size_est_without_init() override {
        // offset + stateNum(int) + maxLen(1 byte) + stateNum bytes of length
        return sizeof(T) + sizeof(int) + 1 + 65536 * sizeof(unsigned char);
    }    

    // perform encoding
    size_t encode(const std::vector<T> &bins, uchar *&bytes) override {
        return encode(bins.data(), bins.size(), bytes);
    }

    size_t encode(const T *bins, size_t num_bin, uchar *&bytes) {
        uchar *out_begin = bytes + sizeof(size_t);  // 留出空间写长度
        uchar *p = out_begin;

        uint64_t bitbuf = 0;     // LSB-first bit buffer
        unsigned nbits  = 0;     // 当前 buffer 中已有的 bit 数（低位开始）

        uint64_t      *code = huffmanTree->code;
        unsigned char *len  = huffmanTree->cout;

        for (size_t i = 0; i < num_bin; ++i) {
            int state = bins[i] - offset;
            // debug 期可以留个保护
            // assert(state >= 0 && static_cast<unsigned>(state) < huffmanTree->stateNum);

            uint64_t c = code[state];   // 低 len[state] bits 有效
            unsigned  l = len[state];

            bitbuf |= (c << nbits);
            nbits  += l;

            // 每有至少 8 个 bit，就吐出一个字节（LSB-first）
            while (nbits >= 8) {
                *p++ = static_cast<uchar>(bitbuf & 0xFFu);
                bitbuf >>= 8;
                nbits  -= 8;
            }
        }

        // 剩余不到 8 bit 的部分，也吐一个字节（高位填 0）
        if (nbits > 0) {
            *p++ = static_cast<uchar>(bitbuf & 0xFFu);
        }

        size_t outSize = static_cast<size_t>(p - out_begin);
        write(outSize, bytes);   // 在 bytes 头部写出 bitstream 的字节数
        bytes += outSize;        // 移到 bitstream 末尾
        return outSize;
    }

    void postprocess_encode() override { Falcons_FreeHuffman(); }

    void preprocess_decode() override {}

    // perform decoding
    /*std::vector<T> decode(const uchar *&bytes, size_t targetLength) override {
        node t = treeRoot;
        std::vector<T> out(targetLength);
        size_t i = 0, byteIndex = 0, count = 0;
        int r;
        node n = treeRoot;
        size_t encodedLength = 0;
        read(encodedLength, bytes);
        if (n->t)  // root->t==1 means that all state values are the same (constant)
        {
            for (count = 0; count < targetLength; count++) out[count] = n->c + offset;
            return out;
        }

        for (i = 0; count < targetLength; i++) {
            byteIndex = i >> 3;  // i/8
            r = i % 8;
            if (((bytes[byteIndex] >> (7 - r)) & 0x01) == 0)
                n = n->left;
            else
                n = n->right;

            if (n->t) {
                out[count] = n->c + offset;
                n = t;
                count++;
            }
        }
        bytes += encodedLength;
        return out;
    }*/
    
        T* decode(const uchar *&bytes, size_t targetLength) override {
        node root = treeRoot;
        // Add SIMD padding to prevent SVE/AVX out-of-bounds reads at buffer tail
        constexpr size_t simd_padding = 64 / sizeof(T);  // 64 bytes = max SVE width (512-bit) / sizeof(T)
        T* out = new T[targetLength + simd_padding];
        size_t count = 0;

        // 先读出 bitstream 的长度（字节数）
        size_t encodedLength = 0;
        read(encodedLength, bytes);

        node n = root;
        if (n->t) {  // 常数树：根就是叶子
            T val = n->c + offset;
            for (size_t i = 0; i < targetLength; ++i) {
                out[i] = val;
            }
            // 跳过 bitstream
            bytes += encodedLength;
            return out;
        }
        const uchar *p     = bytes;
        const uchar *p_end = bytes + encodedLength;
        if (!usingTable) {
            while (p < p_end && count < targetLength) {
                unsigned char byte = *p++;   // 当前字节，LSB-first

                for (int b = 0; b < 8 && count < targetLength; ++b) {
                    int bit = byte & 1u;
                    byte >>= 1;

                    n = bit ? n->right : n->left;

                    if (n->t) {
                        out[count++] = n->c + offset;
                        n = root;
                    }
                }
            }
            bytes = p_end;
            return out;
        }
        else {
        // while (p < p_end && count < targetLength) {
        //     unsigned char byte = *p++;   // 当前字节，LSB-first

        //     for (int b = 0; b < 8 && count < targetLength; ++b) {
        //         int bit = byte & 1u;
        //         byte >>= 1;

        //         n = bit ? n->right : n->left;

        //         if (n->t) {
        //             out[count++] = n->c + offset;
        //             n = root;
        //         }
        //     }
        // }
            uint64_t bitbuf = 0;
            int cur_len = 0;
            const uint64_t L1_mask = (1u << L1) - 1;
            const TableEntry* __restrict__ L1_ptr = L1_table.data();
            // int lenCount[20] = {0};

            bool valid_len1 = n->left->t;
            int center = L1_ptr[0].symbol;

            // auto refill = [&]() __attribute__((always_inline)) {
            //     while (cur_len <= 56 && p < p_end)
            //         bitbuf |= (static_cast<uint64_t>(*p++) << cur_len), cur_len += 8;
            // };

            
            if (!valid_len1) {
                while (count < targetLength) {
                    if (cur_len < L1)
                        while (cur_len <= 56 && p < p_end)
                        bitbuf |= (static_cast<uint64_t>(*p++) << cur_len), cur_len += 8;
                    const TableEntry& entry = L1_ptr[bitbuf & L1_mask];

                    if (entry.is_leaf) {
                        out[count++] = entry.symbol;
                        bitbuf >>= entry.code_len;
                        cur_len -= entry.code_len;
                        // ++lenCount[entry.code_len];
                        continue;
                    }

                    n = root;
                    while (!n->t) {
                        if (cur_len == 0)
                            while (cur_len <= 56 && p < p_end)
                                bitbuf |= (static_cast<uint64_t>(*p++) << cur_len), cur_len += 8;

                        bool bit = bitbuf & 1;
                        bitbuf >>= 1;
                        --cur_len;

                        n = bit ? n->right : n->left;
                    }
                
                    out[count++] = n->c + offset;
                }
            }
            else {
                while (count < targetLength) {
                    if (cur_len < L1)
                        while (cur_len <= 56 && p < p_end)
                            bitbuf |= (static_cast<uint64_t>(*p++) << cur_len), cur_len += 8;
                    if(!(bitbuf & 1u)) {
                        out[count++] = center;
                        bitbuf >>= 1;
                        cur_len -= 1;
                        // ++lenCount[1];
                        continue;
                    }
                    const TableEntry& entry = L1_ptr[bitbuf & L1_mask];

                    if (entry.is_leaf) {
                        out[count++] = entry.symbol;
                        bitbuf >>= entry.code_len;
                        cur_len -= entry.code_len;
                        // ++lenCount[entry.code_len];
                        continue;
                    }

                    n = root;
                    while (!n->t) {
                        if (cur_len == 0)
                            while (cur_len <= 56 && p < p_end)
                                bitbuf |= (static_cast<uint64_t>(*p++) << cur_len), cur_len += 8;

                        bool bit = bitbuf & 1;
                        bitbuf >>= 1;
                        --cur_len;

                        n = bit ? n->right : n->left;
                    }
                
                    out[count++] = n->c + offset;
                }
            }
            // for (int i = 0; i < 20; ++i) {
            //     if (lenCount[i] > 0) {
            //         std::cout << "lenCount[" << i << "] = " << lenCount[i] << " hit ratio: " << lenCount[i] / static_cast<double>(targetLength)<< std::endl;
            //     }
            // }
            bytes = p_end;
        }
        return out;
    }
    
    std::vector<T> decode2(const uchar *&bytes, size_t targetLength) {
        node root = treeRoot;
// #ifdef Falcons_PRINT_TIMINGS
//         Timer timer(true);
// #endif
        std::vector<T> out(targetLength);
// #ifdef Falcons_PRINT_TIMINGS
//         timer.stop("vector allocate time in decoding");
// #endif
        size_t count = 0;

        // 先读出 bitstream 的长度（字节数）
        size_t encodedLength = 0;
        read(encodedLength, bytes);

        node n = root;
        if (n->t) {  // 常数树：根就是叶子
            T val = n->c + offset;
            for (size_t i = 0; i < targetLength; ++i) {
                out[i] = val;
            }
            // 跳过 bitstream
            bytes += encodedLength;
            return out;
        }

        const uchar *p     = bytes;
        const uchar *p_end = bytes + encodedLength;

        while (p < p_end && count < targetLength) {
            unsigned char byte = *p++;   // 当前字节，LSB-first

            // 这个字节里最多有 8 个 bit 可用
            for (int b = 0; b < 8 && count < targetLength; ++b) {
                int bit = byte & 1u;
                byte >>= 1;

                n = bit ? n->right : n->left;

                if (n->t) {
                    out[count++] = n->c + offset;
                    n = root;
                }
            }
        }

        // 消费掉整个 bitstream
        bytes = p_end;
        return out;
    }
    /*
    std::vector<T> decode(const uchar *&bytes, size_t targetLength) override {
       // if (!canonReady) {
       //     throw std::runtime_error("Huffman decode: canonical tables not built");
       // }

        std::vector<T> out(targetLength);
        size_t count = 0;

        // 先读出 bitstream 的长度（字节数）
        size_t encodedLength = 0;
        read(encodedLength, bytes);

        const uchar *p     = bytes;
        const uchar *p_end = bytes + encodedLength;

        uint32_t code   = 0;   // 这里的 code 是“MSB-first”的整数表示
        int      length = 0;   // 当前已累积的 bit 数

        while (p < p_end && count < targetLength) {
            unsigned char byte = *p++;  // 一个字节里 8 个 bit，顺序是 b0,b1,...,b7

            // LSB-first：从 bit0 到 bit7 依次是整个 bitstream 的时间顺序
            for (int b = 0; b < 8 && count < targetLength; ++b) {
                int bit = (byte & 1u);   // 取最低位
                byte >>= 1;              // 右移，准备下一个 bit

                // 按 canonical 的规则累积成 MSB-first 的整数码
                code = (code << 1) | (uint32_t)bit;
                ++length;

                if (length < canonMinLen) {
                    continue;  // 码长还不够，肯定无法匹配任何符号
                }
                if (length > canonMaxLen) {
                    // 理论上不该发生（bitstream 和长度分布必须一致）
                    // 简单恢复一下，避免死循环
                    code   = 0;
                    length = 0;
                    continue;
                }

                int L      = length;
                int firstC = canonFirstCode[L];  // 该长度下的第一个 canonical code
                int cnt    = canonBlCount[L];    // 该长度下 code 的个数
                int diff   = (int)code - firstC;

                if (diff >= 0 && diff < cnt) {
                    // 命中：这个 code 对应某个符号
                    int symbolIndex = canonFirstSymbol[L] + diff;
                    int state       = canonSymbolOrder[symbolIndex];

                    out[count++] = static_cast<T>(state + offset);

                    // reset，准备解析下一个符号
                    code   = 0;
                    length = 0;
                }
                // 否则继续累积更多 bit
            }
        }

        // 消费掉 bitstream
        bytes = p_end;
        return out;
    }
    */
    // empty function
    void postprocess_decode() override { Falcons_FreeHuffman(); }
    /*
    // load Huffman tree
    void load(const uchar *&c, size_t &remaining_length) override {
        read(offset, c, remaining_length);
        nodeCount = bytesToInt32_bigEndian(c);
        int stateNum = bytesToInt32_bigEndian(c + sizeof(int)) * 2;
        size_t encodeStartIndex;
        if (nodeCount <= 256)
            encodeStartIndex = 1 + 3 * nodeCount * sizeof(unsigned char) + nodeCount * sizeof(T);
        else if (nodeCount <= 65536)
            encodeStartIndex =
                1 + 2 * nodeCount * sizeof(unsigned short) + nodeCount * sizeof(unsigned char) + nodeCount * sizeof(T);
        else
            encodeStartIndex =
                1 + 2 * nodeCount * sizeof(unsigned int) + nodeCount * sizeof(unsigned char) + nodeCount * sizeof(T);

        huffmanTree = createHuffmanTree(stateNum);
        treeRoot = reconstruct_HuffTree_from_bytes_anyStates(c + sizeof(int) + sizeof(int), nodeCount);
        c += sizeof(int) + sizeof(int) + encodeStartIndex;
        loaded = true;
    }*/

    void load(const uchar *&c, size_t &remaining_length) override {
        // save() writes a 4-byte int (usingTable in bit31 | offset); read a 4-byte int regardless
        // of T -- reading sizeof(T) bytes (e.g. 2 for uint16) would desync the stream.
        int packed_offset;
        read(packed_offset, c, remaining_length);
        usingTable = (packed_offset >> 31) & 0x1u;
        offset = static_cast<T>(packed_offset & 0x7FFFFFFF);
        // read(usingTable, c, remaining_length);
       // std::cout<<offset<<std::endl;

        // 读 stateNum
        if (remaining_length < sizeof(int)) {
            throw std::runtime_error("Huffman load: insufficient data for stateNum");
        }
        int stateNum = bytesToInt32_bigEndian(c);

       // std::cout<<stateNum<<std::endl;
        c += sizeof(int);
        remaining_length -= sizeof(int);

        // 读 maxLen（其实可以不用，但我们存了就读回来）
        /*
        if (remaining_length < 1) {
            throw std::runtime_error("Huffman load: insufficient data for maxLen");
        }
        unsigned char maxLen = *c++;
        remaining_length -= 1;
        (void)maxLen;  // 我们会重新计算一遍 maxLen，不过你也可以用它做 sanity check
        */

        // 分配 HuffmanTree
        huffmanTree = createHuffmanTree(stateNum);

        // 读每个 state 的码长到 cout
        if (remaining_length < static_cast<size_t>(stateNum)) {
            throw std::runtime_error("Huffman load: insufficient data for code lengths");
        }
        for (int i = 0; i < stateNum; ++i) {
            huffmanTree->cout[i] = *c++;
        }
        remaining_length -= static_cast<size_t>(stateNum);

        // ====== 新增：检查有多少个 state 的长度 > 0 ======
        /*
       */

        if (stateNum == 2) {
            // 只有 0 或 1 个有效码长：
            // 视为“所有数据都是同一个符号”的退化情况
            // 对老的 bitstream（全部为 0）我们默认 state = 0；

            int nonZeroStates = 0;
            int lastState     = -1;
            for (int i = 0; i < stateNum; ++i) {
                if (huffmanTree->cout[i] > 0) {
                    ++nonZeroStates;
                    lastState = i;
                }
            }
            int stateIndex = (nonZeroStates == 1) ? lastState : 0;

            // 重置节点池，只建一个叶子节点作为根
            huffmanTree->n_nodes = 0;
            node leaf = new_node2(static_cast<T>(stateIndex), /*t=*/1);
            leaf->left  = nullptr;
            leaf->right = nullptr;
            treeRoot    = leaf;

            // code/cout 对 decode-by-tree 不重要，但可以给 encode 端一个合理的值
            // （这里随便设一个 1-bit 的 0 码）
            std::memset(huffmanTree->code, 0, huffmanTree->stateNum * sizeof(uint64_t));
            std::memset(huffmanTree->cout, 0, huffmanTree->stateNum * sizeof(unsigned char));
            huffmanTree->cout[stateIndex] = 1;
            huffmanTree->code[stateIndex] = 0;
            huffmanTree->maxBitCount      = 1;

            loaded = true;
            return;
        }



        // 基于码长重建 canonical code + decode 表 + encode 用的 code[]
          rebuildTreeFromCodeLengthsLSB();

        loaded = true;
    }

    bool isLoaded() const { return loaded; }

   private:
    HuffmanTree *huffmanTree = nullptr;
    node treeRoot;
    unsigned int nodeCount = 0;
    uchar sysEndianType;  // 0: little endian, 1: big endian
    bool loaded = false;
    T offset;

     // === Canonical Huffman decode tables ===
  //  std::vector<int> canonSymbolOrder;   // symbols sorted by (length, state)
  //  std::vector<int> canonBlCount;       // bl_count[L]: #codes with length L
  //  std::vector<int> canonFirstCode;     // firstCode[L]: first canonical code of length L (MSB-first)
  //  std::vector<int> canonFirstSymbol;   // firstSymbol[L]: index into canonSymbolOrder
  //  int canonMinLen = 0;
    int canonMaxLen = 0;
   // bool canonReady = false;



    static inline unsigned char reverse8(unsigned char x) {
        x = static_cast<unsigned char>((x & 0xF0u) >> 4 | (x & 0x0Fu) << 4);
        x = static_cast<unsigned char>((x & 0xCCu) >> 2 | (x & 0x33u) << 2);
        x = static_cast<unsigned char>((x & 0xAAu) >> 1 | (x & 0x55u) << 1);
        return x;
    }


    node reconstruct_HuffTree_from_bytes_anyStates(const unsigned char *bytes, uint nodeCount) {
        if (nodeCount <= 256) {
            unsigned char *L = static_cast<unsigned char *>(malloc(nodeCount * sizeof(unsigned char)));
            memset(L, 0, nodeCount * sizeof(unsigned char));
            unsigned char *R = static_cast<unsigned char *>(malloc(nodeCount * sizeof(unsigned char)));
            memset(R, 0, nodeCount * sizeof(unsigned char));
            T *C = static_cast<T *>(malloc(nodeCount * sizeof(T)));
            memset(C, 0, nodeCount * sizeof(T));
            unsigned char *t = static_cast<unsigned char *>(malloc(nodeCount * sizeof(unsigned char)));
            memset(t, 0, nodeCount * sizeof(unsigned char));
            // TODO: Endian type
            // unsigned char cmpSysEndianType = bytes[0];
            // if(cmpSysEndianType!=(unsigned char)sysEndianType)
            // {
            //  unsigned char* p = (unsigned char*)(bytes+1+2*nodeCount*sizeof(unsigned char));
            //  size_t i = 0, size = nodeCount*sizeof(unsigned int);
            //  while(1)
            //  {
            //      symTransform_4bytes(p);
            //      i+=sizeof(unsigned int);
            //      if(i<size)
            //          p+=sizeof(unsigned int);
            //      else
            //          break;
            //  }
            // }
            memcpy(L, bytes + 1, nodeCount * sizeof(unsigned char));
            memcpy(R, bytes + 1 + nodeCount * sizeof(unsigned char), nodeCount * sizeof(unsigned char));
            memcpy(C, bytes + 1 + 2 * nodeCount * sizeof(unsigned char), nodeCount * sizeof(T));
            memcpy(t, bytes + 1 + 2 * nodeCount * sizeof(unsigned char) + nodeCount * sizeof(T),
                   nodeCount * sizeof(unsigned char));
            node root = this->new_node2(C[0], t[0]);
            this->unpad_tree<uchar>(L, R, C, t, 0, root);
            free(L);
            free(R);
            free(C);
            free(t);
            return root;
        } else if (nodeCount <= 65536) {
            unsigned short *L = static_cast<unsigned short *>(malloc(nodeCount * sizeof(unsigned short)));
            memset(L, 0, nodeCount * sizeof(unsigned short));
            unsigned short *R = static_cast<unsigned short *>(malloc(nodeCount * sizeof(unsigned short)));
            memset(R, 0, nodeCount * sizeof(unsigned short));
            T *C = static_cast<T *>(malloc(nodeCount * sizeof(T)));
            memset(C, 0, nodeCount * sizeof(T));
            unsigned char *t = static_cast<unsigned char *>(malloc(nodeCount * sizeof(unsigned char)));
            memset(t, 0, nodeCount * sizeof(unsigned char));

            // TODO: Endian type
            // unsigned char cmpSysEndianType = bytes[0];
            // if(cmpSysEndianType!=(unsigned char)sysEndianType)
            // {
            //  unsigned char* p = (unsigned char*)(bytes+1);
            //  size_t i = 0, size = 3*nodeCount*sizeof(unsigned int);
            //  while(1)
            //  {
            //      symTransform_4bytes(p);
            //      i+=sizeof(unsigned int);
            //      if(i<size)
            //          p+=sizeof(unsigned int);
            //      else
            //          break;
            //  }
            // }

            memcpy(L, bytes + 1, nodeCount * sizeof(unsigned short));
            memcpy(R, bytes + 1 + nodeCount * sizeof(unsigned short), nodeCount * sizeof(unsigned short));
            memcpy(C, bytes + 1 + 2 * nodeCount * sizeof(unsigned short), nodeCount * sizeof(T));

            memcpy(t, bytes + 1 + 2 * nodeCount * sizeof(unsigned short) + nodeCount * sizeof(T),
                   nodeCount * sizeof(unsigned char));

            node root = this->new_node2(0, 0);
            this->unpad_tree<unsigned short>(L, R, C, t, 0, root);
            free(L);
            free(R);
            free(C);
            free(t);
            return root;
        } else  // nodeCount>65536
        {
            unsigned int *L = static_cast<unsigned int *>(malloc(nodeCount * sizeof(unsigned int)));
            memset(L, 0, nodeCount * sizeof(unsigned int));
            unsigned int *R = static_cast<unsigned int *>(malloc(nodeCount * sizeof(unsigned int)));
            memset(R, 0, nodeCount * sizeof(unsigned int));
            T *C = static_cast<T *>(malloc(nodeCount * sizeof(T)));
            memset(C, 0, nodeCount * sizeof(T));
            unsigned char *t = static_cast<unsigned char *>(malloc(nodeCount * sizeof(unsigned char)));
            memset(t, 0, nodeCount * sizeof(unsigned char));
            // TODO: Endian type
            // unsigned char cmpSysEndianType = bytes[0];
            // if(cmpSysEndianType!=(unsigned char)sysEndianType)
            // {
            //  unsigned char* p = (unsigned char*)(bytes+1);
            //  size_t i = 0, size = 3*nodeCount*sizeof(unsigned int);
            //  while(1)
            //  {
            //      symTransform_4bytes(p);
            //      i+=sizeof(unsigned int);
            //      if(i<size)
            //          p+=sizeof(unsigned int);
            //      else
            //          break;
            //  }
            // }

            memcpy(L, bytes + 1, nodeCount * sizeof(unsigned int));
            memcpy(R, bytes + 1 + nodeCount * sizeof(unsigned int), nodeCount * sizeof(unsigned int));
            memcpy(C, bytes + 1 + 2 * nodeCount * sizeof(unsigned int), nodeCount * sizeof(T));

            memcpy(t, bytes + 1 + 2 * nodeCount * sizeof(unsigned int) + nodeCount * sizeof(T),
                   nodeCount * sizeof(unsigned char));

            node root = this->new_node2(0, 0);
            this->unpad_tree<unsigned int>(L, R, C, t, 0, root);
            free(L);
            free(R);
            free(C);
            free(t);
            return root;
        }
    }

    node new_node(size_t freq, T c, node a, node b) {
        node n = huffmanTree->pool + huffmanTree->n_nodes++;
        if (freq) {
            n->c = c;
            n->freq = freq;
            n->t = 1;
            // printf("new_node: c = %d, freq = %zu, t = %d \n", n->c, n->freq, n->t);
        } else {
            n->left = a;
            n->right = b;
            n->freq = a->freq + b->freq;
            n->t = 0;
            // printf("new_node: c = %d, freq = %zu, t = %d, left = %d, right = %d \n", n->c, n->freq, n->t, n->left->c,
            // n->right->c);
            // n->c = 0;
        }
        return n;
    }

    node new_node2(T c, unsigned char t) {
        huffmanTree->pool[huffmanTree->n_nodes].c = c;
        huffmanTree->pool[huffmanTree->n_nodes].t = t;
        return huffmanTree->pool + huffmanTree->n_nodes++;
    }

    /* priority queue */
    void qinsert(node n) {
        int j, i = huffmanTree->qend++;
        while ((j = (i >> 1)))  // j=i/2
        {
            if (huffmanTree->qq[j]->freq <= n->freq) break;
            huffmanTree->qq[i] = huffmanTree->qq[j], i = j;
        }
        huffmanTree->qq[i] = n;
    }

    node qremove() {
        int i = 1, l;
        node n = huffmanTree->qq[i = 1];
        node p;
        if (huffmanTree->qend < 2) return nullptr;
        huffmanTree->qend--;
        huffmanTree->qq[i] = huffmanTree->qq[huffmanTree->qend];

        while ((l = (i << 1)) < huffmanTree->qend) {  // l=(i*2)
            if (l + 1 < huffmanTree->qend && huffmanTree->qq[l + 1]->freq < huffmanTree->qq[l]->freq) l++;
            if (huffmanTree->qq[i]->freq > huffmanTree->qq[l]->freq) {
                p = huffmanTree->qq[i];
                huffmanTree->qq[i] = huffmanTree->qq[l];
                huffmanTree->qq[l] = p;
                i = l;
            } else {
                break;
            }
        }
        return n;
    }

    /* walk the tree and put 0s and 1s */
    /**
     * @out1 should be set to 0.
     * @out2 should be 0 as well.
     * @index: the index of the byte
     * */
    void build_code(node n, uint64_t code_val, int len) {
        if (n->t) {
            // 叶子：记录码字和长度
            assert(len <= 64);
            huffmanTree->code[n->c] = code_val;  // 低 len bits 有效
            huffmanTree->cout[n->c] = static_cast<unsigned char>(len);
            if (len > huffmanTree->maxBitCount) {
                huffmanTree->maxBitCount = len;
            }
            return;
        }

        // 左子树：追加一个 0 bit（code_val 不变），长度+1
        build_code(n->left, code_val, len + 1);

        // 右子树：在第 len 位上置 1，再长度+1
        uint64_t code_right = code_val | (uint64_t(1) << len);
        build_code(n->right, code_right, len + 1);
    }

    


    void buildCanonicalCode(size_t* frequencies, size_t length) {
        if (!huffmanTree) return;

        const int n = static_cast<int>(huffmanTree->stateNum);

        // 1. 收集所有出现过的 state（cout[state] > 0）
        std::vector<int> symbols;
        symbols.reserve(n);
        int maxLen = 0;
        for (int s = 0; s < n; ++s) {
            unsigned char L = huffmanTree->cout[s];
            if (L > 0) {
                symbols.push_back(s);
                if (L > maxLen) maxLen = static_cast<int>(L);
            }
        }
        if (symbols.empty()) return;  // safety

        // 2. 按 (len, symbol) 排序
        std::sort(symbols.begin(), symbols.end(),
                  [&](int a, int b) {
                      unsigned char la = huffmanTree->cout[a];
                      unsigned char lb = huffmanTree->cout[b];
                      if (la != lb) return la < lb;
                      return a < b;
                  });

        // 3. 统计每个长度的数量 bl_count[L]
        std::vector<int> bl_count(maxLen + 1, 0);
        std::vector<uint32_t> lengthFreqSum(maxLen + 1, 0);
        for (int s : symbols) {
            unsigned char L = huffmanTree->cout[s];
            ++bl_count[L];
            lengthFreqSum[L] += frequencies[s + offset];
        }

        // 4. 计算各长度的第一个 canonical code（MSB-first）
        std::vector<uint32_t> next_code(maxLen + 1, 0);
        uint32_t code = 0;
        bl_count[0] = 0;
        for (int bits = 1; bits <= maxLen; ++bits) {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        
        size_t tableFreq = 0;
        for (int i = 2; i <= maxLen && i <= L1; ++i) {
            tableFreq += lengthFreqSum[i];
        }

        // for (int i = 1; i <= maxLen; ++i) {
        //     std::cout << "lengthFreqSum[" << i << "] = " << lengthFreqSum[i] << '\n';
        // }
        if (length - tableFreq -  lengthFreqSum[1] >= tableFreq * 0.88 || static_cast<double>(lengthFreqSum[1]) / length >= 0.98) {
            usingTable = false;
        }
        else {
            usingTable = true;
        }
        // std::cout << "tableFreq = " << tableFreq << ", length = " << length << ", lengthFreqSum[1] = " << lengthFreqSum[1] << '\n';
        // std::cout << "usingTable = " << usingTable << '\n';
        // usingTable = 1;

        // 5. 为每个 symbol 分配 canonical 码字，并转为 LSB-first 存储
        // auto reverse_len_bits = [](uint32_t c, int len) -> uint32_t {
        //     uint32_t r = 0;
        //     for (int i = 0; i < len; ++i) {
        //         if ((c >> (len - 1 - i)) & 1u) {
        //             r |= (1u << i);
        //         }
        //     }
        //     return r;
        // };

        auto reverse_len_bits = [](uint32_t c, int len) -> uint32_t {
            c = ((c & 0x55555555u) << 1) | ((c >> 1) & 0x55555555u);
            c = ((c & 0x33333333u) << 2) | ((c >> 2) & 0x33333333u);
            c = ((c & 0x0F0F0F0Fu) << 4) | ((c >> 4) & 0x0F0F0F0Fu);
            c = ((c & 0x00FF00FFu) << 8) | ((c >> 8) & 0x00FF00FFu);
            c = (c << 16) | (c >> 16);
            return c >> (32 - len);
        };

        for (int s : symbols) {
            int len = static_cast<int>(huffmanTree->cout[s]);
            uint32_t msb_code = next_code[len]++;
            uint32_t lsb_code = reverse_len_bits(msb_code, len);
            huffmanTree->code[s] = static_cast<uint64_t>(lsb_code);
        }

        // maxBitCount 直接用 maxLen 即可（和原始 build_code 的 maxBitCount 一致）
        huffmanTree->maxBitCount = maxLen;
       // canonMaxLen = maxLen;
        //canonReady = true;
    }


    /**
     * Compute the frequency of the data and build the Huffman tree
     * @param HuffmanTree* huffmanTree (output)
     * @param int *s (input)
     * @param size_t length (input)
     * */
    void init(const T *s, size_t length) {
        T max = s[0];
        offset = 0;  // offset is min
/*
#if (Falcons_USE_SKA_HASH) && (INTPTR_MAX == INT64_MAX)  // use ska for 64bit system
        ska::unordered_map<T, size_t> frequency;
#else   // most likely 32bit system
        std::unordered_map<T, size_t> frequency;
#endif  // INTPTR_MAX == INT64_MAX
*/

/*
        for (const auto &kv : frequency) {
            auto k = kv.first;
            if (k > max) {
                max = k;
            }
            if (k < offset) {
                offset = k;
            }
        }
*/  
        //Timer timer(true);
        const size_t ui16_range= 1<<16;
        std::vector<size_t> frequencyList(ui16_range, 0);
        auto frenqencies = frequencyList.data();
        for (size_t i = 0; i < length; i++) {
            /*
            auto k = s[i];
            if (k > max) {
                max = k;
            }
            if (k < offset) {
                offset = k;
            }*/
            //assert(s[i]>0 && s[i]<ui16_range);
            frenqencies[s[i]] += 1;
        }
        for (size_t i = 0; i < ui16_range; i++) {
            if (frenqencies[i] != 0) {
                offset = i;
                break;
            }

        }
        for (size_t i = ui16_range - 1; i >= 0 ; i--) {
            if (frenqencies[i] != 0) {
                max = i;
                break;
            }

        }
       // std::cout<<offset<<" "<<max<<std::endl;


        int stateNum = max - offset + 2;
       // auto tid = omp_get_thread_num();
       //  #pragma omp critical
       // std::cout<<tid<<" statenum "<< stateNum<<std::endl;
        //timer.stop("count");
        huffmanTree = createHuffmanTree(stateNum);
        // to produce the same huffman three on linux & win, we need to iterate through ordered_map in a fixed order
        
        for (int i = offset; i <= max; i++) {
            if (frenqencies[i] != 0) {
                qinsert(new_node(frenqencies[i], i - offset, nullptr, nullptr));
            }
        }
        // for (const auto &f : frequency) {
        //     qinsert(new_node(f.second, f.first - offset, nullptr, nullptr));
        // }

        while (huffmanTree->qend > 2) {
            auto left = qremove();
            auto right = qremove();
            qinsert(new_node(0, 0, left, right));
        }


        build_code(huffmanTree->qq[1], 0ULL, 0);
        treeRoot = huffmanTree->qq[1];
        buildCanonicalCode(frenqencies, length);

    }

    void init(const T *s, size_t length, size_t* frequencies) {
        T max = s[0];
        offset = 0;  // offset is min
        const size_t ui16_range= 1<<16;
        for (size_t i = 0; i < ui16_range; i++) {
            if (frequencies[i] != 0) {
                offset = i;
                break;
            }

        }
        for (size_t i = ui16_range - 1; i >= 0 ; i--) {
            if (frequencies[i] != 0) {
                max = i;
                break;
            }

            
        }
        int stateNum = max - offset + 2;
        //std::cout << "offset: " << offset << ", max: " << max << ", length: " << length << std::endl;
        huffmanTree = createHuffmanTree(stateNum);
        // to produce the same huffman three on linux & win, we need to iterate through ordered_map in a fixed order
        
        for (int i = offset; i <= max; i++) {
            if (frequencies[i] != 0) {
                qinsert(new_node(frequencies[i], i - offset, nullptr, nullptr));
            }
        }
        // for (const auto &f : frequency) {
        //     qinsert(new_node(f.second, f.first - offset, nullptr, nullptr));
        // }

        while (huffmanTree->qend > 2) {
            auto left = qremove();
            auto right = qremove();
            qinsert(new_node(0, 0, left, right));
        }


        build_code(huffmanTree->qq[1], 0ULL, 0);
        treeRoot = huffmanTree->qq[1];
        buildCanonicalCode(frequencies, length);
    }
    
    template <class T1>
    void pad_tree(T1 *L, T1 *R, T *C, unsigned char *t, unsigned int i, node root) {
        C[i] = root->c;
        t[i] = root->t;
        node lroot = root->left;
        if (lroot != nullptr) {
            huffmanTree->n_inode++;
            L[i] = huffmanTree->n_inode;
            pad_tree(L, R, C, t, huffmanTree->n_inode, lroot);
        }
        node rroot = root->right;
        if (rroot != nullptr) {
            huffmanTree->n_inode++;
            R[i] = huffmanTree->n_inode;
            pad_tree(L, R, C, t, huffmanTree->n_inode, rroot);
        }
    }

    template <class T1>
    void unpad_tree(T1 *L, T1 *R, T *C, unsigned char *t, unsigned int i, node root) {
        // root->c = C[i];
        if (root->t == 0) {
            T1 l, r;
            l = L[i];
            if (l != 0) {
                node lroot = new_node2(C[l], t[l]);
                root->left = lroot;
                unpad_tree(L, R, C, t, l, lroot);
            }
            r = R[i];
            if (r != 0) {
                node rroot = new_node2(C[r], t[r]);
                root->right = rroot;
                unpad_tree(L, R, C, t, r, rroot);
            }
        }
    }
    /*
    void buildCanonicalFromLengths() {
        if (!huffmanTree) return;

        const int n = static_cast<int>(huffmanTree->stateNum);

        // 1. 收集所有出现过的 state（长度>0）
        std::vector<int> symbols;
        symbols.reserve(n);
        int maxLen = 0;
        int minLen = 0x7fffffff;

        for (int s = 0; s < n; ++s) {
            unsigned char L = huffmanTree->cout[s];
            if (L > 0) {
                symbols.push_back(s);
                if (L > maxLen) maxLen = (int)L;
                if (L < minLen) minLen = (int)L;
            }
        }

        if (symbols.empty()) {
            canonReady = false;
            canonMinLen = canonMaxLen = 0;
            return;
        }

        canonMinLen = minLen;
        canonMaxLen = maxLen;

        // 2. 按 (len, state) 排序
        std::sort(symbols.begin(), symbols.end(),
                  [&](int a, int b) {
                      unsigned char la = huffmanTree->cout[a];
                      unsigned char lb = huffmanTree->cout[b];
                      if (la != lb) return la < lb;
                      return a < b;
                  });

        canonSymbolOrder = symbols;

        // 3. 统计每个长度的数量 bl_count[L]
        canonBlCount.assign(maxLen + 1, 0);
        for (int s : symbols) {
            unsigned char L = huffmanTree->cout[s];
            ++canonBlCount[(int)L];
        }

        // 4. 计算各长度的第一个 canonical code（MSB-first）
        canonFirstCode.assign(maxLen + 1, 0);
        canonFirstSymbol.assign(maxLen + 1, 0);

        int code = 0;
        canonBlCount[0] = 0;
        for (int bits = 1; bits <= maxLen; ++bits) {
            code = (code + canonBlCount[bits - 1]) << 1;
            canonFirstCode[bits] = code;
        }

        // firstSymbol[L]：在 symbols[] 中，长度为 L 的第一个位置
        int sum = 0;
        for (int bits = 1; bits <= maxLen; ++bits) {
            canonFirstSymbol[bits] = sum;
            sum += canonBlCount[bits];
        }

        // 5. 为 encode 生成 LSB-first 码字（可选，如果你还想用 huffmanTree->code 做 LSB-first encode）
        auto reverse_len_bits = [](uint32_t c, int len) -> uint32_t {
            uint32_t r = 0;
            for (int i = 0; i < len; ++i) {
                if ((c >> (len - 1 - i)) & 1u) {
                    r |= (1u << i);
                }
            }
            return r;
        };

        // next_code[L]：当前长度 L 下一个可用 canonical code（MSB-first）
        std::vector<int> next_code(maxLen + 1, 0);
        for (int bits = 1; bits <= maxLen; ++bits) {
            next_code[bits] = canonFirstCode[bits];
        }

        for (int s : symbols) {
            int len = (int)huffmanTree->cout[s];
            int msb_code = next_code[len]++;

            // encode 若用 LSB-first，可把 MSB-first 反转后写入 huffmanTree->code
            uint32_t lsb_code = reverse_len_bits((uint32_t)msb_code, len);
            huffmanTree->code[s] = (uint64_t)lsb_code;
        }

        huffmanTree->maxBitCount = maxLen;
        canonReady = true;
    }*/

      void rebuildTreeFromCodeLengthsLSB() {
        if (!huffmanTree) return;

        const int n = static_cast<int>(huffmanTree->stateNum);

        // 1. 收集所有用到的 state（length > 0），并统计 min/maxLen
        std::vector<int> symbols;
        symbols.reserve(n);
        int maxLen = 0;
        for (int s = 0; s < n; ++s) {
            unsigned char L = huffmanTree->cout[s];
            if (L > 0) {
                symbols.push_back(s);
                if (L > maxLen) maxLen = static_cast<int>(L);
            }
        }
        if (symbols.empty()) {
            treeRoot = nullptr;
            return;
        }

        // 2. canonical 排序：先按长度，再按 state index
        std::sort(symbols.begin(), symbols.end(),
                  [&](int a, int b) {
                      unsigned char la = huffmanTree->cout[a];
                      unsigned char lb = huffmanTree->cout[b];
                      if (la != lb) return la < lb;
                      return a < b;
                  });

        // 3. 统计每个长度的数量 bl_count[L]
        std::vector<int> bl_count(maxLen + 1, 0);
        for (int s : symbols) {
            unsigned char L = huffmanTree->cout[s];
            ++bl_count[static_cast<int>(L)];
        }

        // 4. 计算各长度的第一个 canonical code（MSB-first 整数）
        std::vector<int> next_code(maxLen + 1, 0);
        int code = 0;
        bl_count[0] = 0;
        for (int bits = 1; bits <= maxLen; ++bits) {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
            // std::cout << "length " << bits - 1<< " has " << bl_count[bits];
        }
        // std::cout << std::endl;
        // 5. 工具：把 MSB-first 的 canonical 整数码反转成 LSB-first

        auto reverse_len_bits = [](uint32_t c, int len) -> uint32_t {
            c = ((c & 0x55555555u) << 1) | ((c >> 1) & 0x55555555u);
            c = ((c & 0x33333333u) << 2) | ((c >> 2) & 0x33333333u);
            c = ((c & 0x0F0F0F0Fu) << 4) | ((c >> 4) & 0x0F0F0F0Fu);
            c = ((c & 0x00FF00FFu) << 8) | ((c >> 8) & 0x00FF00FFu);
            c = (c << 16) | (c >> 16);
            return c >> (32 - len);
        };

        // 6. 清空 node 池，构造根节点
        huffmanTree->n_nodes = 0;
        node root = new_node2(0, 0);  // t=0, c 暂时无意义
        root->left  = nullptr;
        root->right = nullptr;
        treeRoot = root;
        if(usingTable) {
            L1_table.resize(1 << L1);
            for (int s : symbols) {
                int len = static_cast<int>(huffmanTree->cout[s]);
                int msb_code = next_code[len]++;            // canonical MSB-first code
                uint32_t lsb_code = reverse_len_bits(static_cast<uint32_t>(msb_code), len);
                huffmanTree->code[s] = static_cast<uint64_t>(lsb_code);

                node cur = root;
                for (int b = 0; b < len; ++b) {
                    int bit = (lsb_code >> b) & 1;
                    node &child = bit ? cur->right : cur->left;
                    if (!child) {
                        child = new_node2(0, 0);
                        child->left  = nullptr;
                        child->right = nullptr;
                    }
                    cur = child;
                }
                // 走完 len 个 bit，cur 即为叶子节点
                cur->t = 1;
                cur->c = static_cast<T>(s);


                if (len <= L1) {
                    // --len;
                    // lsb_code >>= 1;
                    int reps = 1 << (L1 - len);
                    for (int r = 0; r < reps; ++r) {
                        uint32_t idx = lsb_code | (static_cast<uint32_t>(r) << len);
                        L1_table[idx] = {s + offset, len, true};
                    }
                }
            }
        }
        else {
            for (int s : symbols) {
                int len = static_cast<int>(huffmanTree->cout[s]);
                int msb_code = next_code[len]++;            // canonical MSB-first code
                uint32_t lsb_code = reverse_len_bits(static_cast<uint32_t>(msb_code), len);
                huffmanTree->code[s] = static_cast<uint64_t>(lsb_code);

                node cur = root;
                for (int b = 0; b < len; ++b) {
                    int bit = (lsb_code >> b) & 1;
                    node &child = bit ? cur->right : cur->left;
                    if (!child) {
                        child = new_node2(0, 0);
                        child->left  = nullptr;
                        child->right = nullptr;
                    }
                    cur = child;
                }
                // 走完 len 个 bit，cur 即为叶子节点
                cur->t = 1;
                cur->c = static_cast<T>(s);
            }

        }  

    }


    template <class T1>
    unsigned int convert_HuffTree_to_bytes_anyStates(unsigned int nodeCount, unsigned char *out) {
        T1 *L = static_cast<T1 *>(malloc(nodeCount * sizeof(T1)));
        memset(L, 0, nodeCount * sizeof(T1));
        T1 *R = static_cast<T1 *>(malloc(nodeCount * sizeof(T1)));
        memset(R, 0, nodeCount * sizeof(T1));
        T *C = static_cast<T *>(malloc(nodeCount * sizeof(T)));
        memset(C, 0, nodeCount * sizeof(T));
        unsigned char *t = static_cast<unsigned char *>(malloc(nodeCount * sizeof(unsigned char)));
        memset(t, 0, nodeCount * sizeof(unsigned char));

        pad_tree(L, R, C, t, 0, huffmanTree->qq[1]);

        unsigned int totalSize =
            1 + 2 * nodeCount * sizeof(T1) + nodeCount * sizeof(unsigned char) + nodeCount * sizeof(T);
        //*out = (unsigned char*)malloc(totalSize);
        out[0] = sysEndianType;
        memcpy(out + 1, L, nodeCount * sizeof(T1));
        memcpy(out + 1 + nodeCount * sizeof(T1), R, nodeCount * sizeof(T1));
        memcpy(out + 1 + 2 * nodeCount * sizeof(T1), C, nodeCount * sizeof(T));
        memcpy(out + 1 + 2 * nodeCount * sizeof(T1) + nodeCount * sizeof(T), t, nodeCount * sizeof(unsigned char));

        free(L);
        free(R);
        free(C);
        free(t);
        return totalSize;
    }

    void Falcons_FreeHuffman() {
    if (huffmanTree != nullptr) {
        free(huffmanTree->pool);
        huffmanTree->pool = nullptr;

        free(huffmanTree->qqq);
        huffmanTree->qqq = nullptr;

        if (huffmanTree->code != nullptr) {
            free(huffmanTree->code);
            huffmanTree->code = nullptr;
        }
        if (huffmanTree->cout != nullptr) {
            free(huffmanTree->cout);
            huffmanTree->cout = nullptr;
        }

        free(huffmanTree);
        huffmanTree = nullptr;
    }
}
};
}  // namespace Falcons

#endif