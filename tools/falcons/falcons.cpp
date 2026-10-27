#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Falcons/api/falcons.hpp"

#define Falcons_FLOAT 0
#define Falcons_DOUBLE 1
#define Falcons_UINT8 2
#define Falcons_INT8 3
#define Falcons_UINT16 4
#define Falcons_INT16 5
#define Falcons_UINT32 6
#define Falcons_INT32 7
#define Falcons_UINT64 8
#define Falcons_INT64 9

inline void usage() {
    printf("Note: Falcons command line arguments are backward compatible with SZ2, \n");
    printf("      use -h2 to show the supported SZ2 command line arguments. \n");
    printf("Usage: falcons <options>\n");
    printf("Options:\n");
    printf("* general options:\n");
    printf("	-h: print the help information\n");
    printf("	-h2: print the help information for SZ2 style command line\n");
    printf("	-v: print the version number\n");
    printf("	-P: enable the OpenMP mode\n");
    printf("	-a : print compression results such as distortions\n");
    printf("* input and output:\n");
    printf("	-i <path> : original input file in binary format\n");
    printf("	-o <path> : decompressed file in binary format\n");
    printf("	-z <path> : compressed file\n");
    printf("	-t : store decompressed file in text format\n");
    //    printf("	-p: print meta data (configuration info)\n");
    printf("* data type:\n");
    printf("	-f: single precision (float type)\n");
    printf("	-d: double precision (double type)\n");
    printf("	-I <width>: signed integer type (width = 16, 32, or 64)\n");
    printf("	-U <width>: unsigned integer type (width = 16, 32, or 64)\n");
    printf("* configuration file: \n");
    printf("	-c <configuration file> : configuration file falcons.config\n");
    printf("* error control: (the error control parameters here will overwrite the setting in falcons.config)\n");
    printf("	-M <error control mode> <error bound (optional)> \n");
    printf("	error control mode as follows: \n");
    printf("		ABS (absolute error bound)\n");
    printf("		REL (value range based error bound, so a.k.a., VR_REL)\n");
    printf("		PSNR (peak signal-to-noise ratio)\n");
    printf("		NORM (norm2 error : sqrt(sum(xi-xi')^2)\n");
    printf("		ABS_AND_REL (using min{ABS, REL})\n");
    printf("		ABS_OR_REL (using max{ABS, REL})\n");
    printf(
        "	error bound can be set directly after the error control mode, or separately with the following "
        "options:\n");
    printf("		-A <absolute error bound>: specifying absolute error bound\n");
    printf("		-R <value_range based relative error bound>: specifying relative error bound\n");
    //    printf("		-P <point-wise relative error bound>: specifying point-wise relative error bound\n");
    printf("		-S <PSNR>: specifying PSNR\n");
    printf("		-N <normErr>: specifying normErr\n");
    printf("* dimensions: \n");
    printf("	-1 <nx> : dimension for 1D data such as data[nx]\n");
    printf("	-2 <nx> <ny> : dimensions for 2D data such as data[ny][nx]\n");
    printf("	-3 <nx> <ny> <nz> : dimensions for 3D data such as data[nz][ny][nx] \n");
    printf("	-4 <nx> <ny> <nz> <np>: dimensions for 4D data such as data[np][nz][ny][nx] \n");
    printf("* examples: \n");
    printf("	falcons -f -i test.dat    -z test.dat.falcons     -3 8 8 128 -M ABS 1e-3 \n");
    printf("	falcons -f -z test.dat.falcons -o test.dat.falcons.out -3 8 8 128 -M REL 1e-3 -a \n");
    printf("	falcons -f -i test.dat    -o test.dat.falcons.out -3 8 8 128 -M ABS_AND_REL -A 1 -R 1e-3 -a \n");
    printf("	falcons -f -i test.dat    -o test.dat.falcons.out -3 8 8 128 -c falcons.config \n");
    printf("	falcons -f -i test.dat    -o test.dat.falcons.out -3 8 8 128 -c falcons.config -M ABS 1e-3 -a\n");
    exit(0);
}

inline void usage_legacy() {
    printf("Note: below are the supported command line arguments in SZ2 style\n");
    printf("Usage: falcons <options>\n");
    printf("Options:\n");
    printf("* operation type:\n");
    printf("	-z <compressed file>: the compression operation with an optionally specified output file.\n");
    printf("                          (the compressed file will be named as <input_file>.falcons if not specified)\n");
    printf("	-x <decompressed file>: the decompression operation with an optionally specified output file\n");
    printf("                      (the decompressed file will be named as <cmpred_file>.out if not specified)\n");
    printf("	-p: print meta data (configuration info)\n");
    printf("	-h: print the help information\n");
    printf("	-v: print the version number\n");
    printf("* data type:\n");
    printf("	-f: single precision (float type)\n");
    printf("	-d: double precision (double type)\n");
    printf("	-I <width>: signed integer type (width = 16, 32, or 64)\n");
    printf("	-U <width>: unsigned integer type (width = 16, 32, or 64)\n");
    printf("* configuration file: \n");
    printf("	-c <configuration file> : configuration file falcons.config\n");
    printf("* error control: (the error control parameters here will overwrite the setting in falcons.config)\n");
    printf("	-M <error bound mode> : 10 options as follows. \n");
    printf("		ABS (absolute error bound)\n");
    printf("		REL (value range based error bound, so a.k.a., VR_REL)\n");
    printf("		ABS_AND_REL (using min{ABS, REL})\n");
    printf("		ABS_OR_REL (using max{ABS, REL})\n");
    printf("		PSNR (peak signal-to-noise ratio)\n");
    printf("		NORM (norm2 error : sqrt(sum(xi-xi')^2)\n");
    //    printf("		PW_REL (point-wise relative error bound)\n");
    printf("	-A <absolute error bound>: specifying absolute error bound\n");
    printf("	-R <value_range based relative error bound>: specifying relative error bound\n");
    //    printf("	-P <point-wise relative error bound>: specifying point-wise relative error bound\n");
    printf("	-S <PSNR>: specifying PSNR\n");
    printf("	-N <normErr>: specifying normErr\n");
    printf("* input data file:\n");
    printf("	-i <original data file> : original data file\n");
    printf("	-s <compressed data file> : compressed data file in decompression\n");
    printf("* output type of decompressed file: \n");
    printf("	-b (by default) : decompressed file stored in binary format\n");
    printf("	-t : decompreadded file stored in text format\n");
    //    printf("	-T : pre-processing with Tucker Tensor Decomposition\n");
    printf("* dimensions: \n");
    printf("	-1 <nx> : dimension for 1D data such as data[nx]\n");
    printf("	-2 <nx> <ny> : dimensions for 2D data such as data[ny][nx]\n");
    printf("	-3 <nx> <ny> <nz> : dimensions for 3D data such as data[nz][ny][nx] \n");
    printf("	-4 <nx> <ny> <nz> <np>: dimensions for 4D data such as data[np][nz][ny][nx] \n");
    printf("* print compression results: \n");
    printf("	-a : print compression results such as distortions\n");
    printf("* examples: \n");
    printf("	falcons -z -f -c falcons.config -i testdata/x86/testfloat_8_8_128.dat -3 8 8 128\n");
    printf("	falcons -z -f -c falcons.config -M ABS -A 1E-3 -i testdata/x86/testfloat_8_8_128.dat -3 8 8 128\n");
    printf("	falcons -x -f -s testdata/x86/testfloat_8_8_128.dat.falcons -3 8 8 128\n");
    printf(
        "	falcons -x -f -s testdata/x86/testfloat_8_8_128.dat.falcons -i testdata/x86/testfloat_8_8_128.dat -3 8 8 128 "
        "-a\n");
    printf("	falcons -z -d -c falcons.config -i testdata/x86/testdouble_8_8_128.dat -3 8 8 128\n");
    printf("	falcons -x -d -s testdata/x86/testdouble_8_8_128.dat.falcons -3 8 8 128\n");
    printf("	falcons -p -s testdata/x86/testdouble_8_8_128.dat.falcons\n");
    exit(0);
}

template <class T>
void compress(char *inPath, char *cmpPath, Falcons::Config &conf) {
    T *data = new T[conf.num];
    Falcons::readfile<T>(inPath, conf.num, data);
    size_t bytesCap = 2 * conf.num * sizeof(T);
    auto bytes = new char[bytesCap];

    Falcons::Timer timer(true);
    size_t outSize = Falcons_compress<T>(conf, data, bytes, bytesCap);
    double compress_time = timer.stop();

    char outputFilePath[1024];
    if (cmpPath == nullptr) {
        snprintf(outputFilePath, 1024, "%s.falcons", inPath);
    } else {
        strcpy(outputFilePath, cmpPath);
    }
    Falcons::writefile(outputFilePath, bytes, outSize);

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", outputFilePath);

    delete[] data;
    delete[] bytes;
}

template <class T>
void decompress(char *inPath, char *cmpPath, char *decPath, Falcons::Config &conf, int binaryOutput, int printCmpResults) {
    size_t cmpSize;
    auto cmpData = Falcons::readfile<char>(cmpPath, cmpSize);

    Falcons::Timer timer(true);
    auto decData = Falcons_decompress<T>(conf, cmpData.get(), cmpSize);
    double compress_time = timer.stop();

    char outputFilePath[1024];
    if (decPath == nullptr) {
        snprintf(outputFilePath, 1024, "%s.out", cmpPath);
    } else {
        strcpy(outputFilePath, decPath);
    }
    if (binaryOutput == 1) {
        Falcons::writefile<T>(outputFilePath, decData, conf.num);
    } else {
        Falcons::writeTextFile<T>(outputFilePath, decData, conf.num);
    }
    if (printCmpResults) {
        // compute the distortion / compression errors...
        size_t totalNbEle;
        auto ori_data = Falcons::readfile<T>(inPath, totalNbEle);
        assert(totalNbEle == conf.num);
        Falcons::verify<T>(ori_data.get(), decData, conf.num);
    }
    delete[] decData;

    printf("compression ratio = %f\n", conf.num * sizeof(T) * 1.0 / cmpSize);
    printf("decompression time = %f seconds.\n", compress_time);
    printf("decompressed file = %s\n", outputFilePath);
}

int main(int argc, char *argv[]) {
    bool binaryOutput = true;
    int printCmpResults = 0;
    int printMeta = 0;
    bool compression = false;
    bool decompression = false;
    int dataType = Falcons_FLOAT;
    char *inPath = nullptr;
    char *cmpPath = nullptr;
    char *conPath = nullptr;
    char *decPath = nullptr;
    bool delCmpPath = false;

    char *errBoundMode = nullptr;
    char *errBound = nullptr;
    char *absErrorBound = nullptr;
    char *relErrorBound = nullptr;
    // char *pwrErrorBound = nullptr;
    char *psnrErrorBound = nullptr;
    char *normErrorBound = nullptr;

    bool legacy_mode = false;
    bool enable_openmp = false;
    size_t r4 = 0;
    size_t r3 = 0;
    size_t r2 = 0;
    size_t r1 = 0;

    int i = 0;
    // int status;
    if (argc == 1) usage();
    int width = -1;

    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-' || argv[i][2]) {
            if (argv[i][1] == 'h' && argv[i][2] == '2') {
                usage_legacy();
            } else {
                usage();
            }
        }
        switch (argv[i][1]) {
            case 'h':
                usage();
                exit(0);
            case 'v':
                printf("Falcons Version: %s\n", Falcons_VER);
                printf("Falcons Data Format Version: %s\n", Falcons_DATA_VER);
                printf("\nThird-party libraries copyright notices:\n");
                printf("----------------------------------------\n");
                printf("ska_hash:\n");
                printf("  Copyright (c) 2017 Malte Skarupke\n");
                printf("  Licensed under the Boost Software License - Version 1.0\n");
                exit(0);
            case 'b':
                binaryOutput = true;
                break;
            case 't':
                binaryOutput = false;
                break;
            case 'a':
                printCmpResults = 1;
                break;
            case 'p':
                printMeta = 1;
                break;
            case 'z':
                compression = true;
                if (i + 1 < argc) {
                    cmpPath = argv[i + 1];
                    if (cmpPath[0] != '-')
                        i++;
                    else
                        cmpPath = nullptr;
                }
                break;
            case 'x':
                legacy_mode = true;
                decompression = true;
                if (i + 1 < argc) {
                    decPath = argv[i + 1];
                    if (decPath[0] != '-')
                        i++;
                    else
                        decPath = nullptr;
                }
                break;
            case 'f':
                dataType = Falcons_FLOAT;
                break;
            case 'd':
                dataType = Falcons_DOUBLE;
                break;
            case 'I':
                if (++i == argc || sscanf(argv[i], "%d", &width) != 1) {
                    usage();
                }
                if (width == 16) {
                    dataType = Falcons_INT16;
                } else if (width == 32) {
                    dataType = Falcons_INT32;
                } else if (width == 64) {
                    dataType = Falcons_INT64;
                } else {
                    usage();
                }
                break;
            case 'U':
                if (++i == argc || sscanf(argv[i], "%d", &width) != 1) {
                    usage();
                }
                if (width == 16) {
                    dataType = Falcons_UINT16;
                } else if (width == 32) {
                    dataType = Falcons_UINT32;
                } else if (width == 64) {
                    dataType = Falcons_UINT64;
                } else {
                    usage();
                }
                break;
            case 'i':
                if (++i == argc) usage();
                inPath = argv[i];
                break;
            case 'o':
                if (++i == argc) usage();
                decPath = argv[i];
                break;
            case 's':
                legacy_mode = true;
                if (++i == argc) usage();
                cmpPath = argv[i];
                break;
            case 'c':
                if (++i == argc) usage();
                conPath = argv[i];
                break;
            case 'P':
                enable_openmp = true;
                break;
            case '1':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1) usage();
                break;
            case '2':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r2) != 1)
                    usage();
                break;
            case '3':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                    sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1)
                    usage();
                break;
            case '4':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                    sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1 ||
                    ++i == argc || sscanf(argv[i], "%zu", &r4) != 1)
                    usage();
                break;
            case 'M':
                if (++i == argc) usage();
                errBoundMode = argv[i];
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    errBound = argv[++i];
                }
                break;
            case 'A':
                if (++i == argc) usage();
                absErrorBound = argv[i];
                break;
            case 'R':
                if (++i == argc) usage();
                relErrorBound = argv[i];
                break;
                //            case 'P':
                //                if (++i == argc)
                //                    usage();
                //                pwrErrorBound = argv[i];
                //                break;
            case 'N':
                if (++i == argc) usage();
                normErrorBound = argv[i];
                break;
            case 'S':
                if (++i == argc) usage();
                psnrErrorBound = argv[i];
                break;
            default:
                usage();
                break;
        }
    }

    if ((inPath == nullptr) && (cmpPath == nullptr)) {
        printf("Error: you need to specify either a raw binary data file or a compressed data file as input\n");
        usage();
        exit(0);
    }

    if (!legacy_mode && inPath != nullptr && cmpPath != nullptr) {
        compression = true;
    }
    if (cmpPath != nullptr && decPath != nullptr) {
        decompression = true;
    }
    char cmpPathTmp[1024];
    if (inPath != nullptr && cmpPath == nullptr && decPath != nullptr) {
        compression = true;
        decompression = true;
        snprintf(cmpPathTmp, 1024, "%s.falcons.tmp", inPath);
        cmpPath = cmpPathTmp;
        delCmpPath = true;
    }
    if (inPath == nullptr || (errBoundMode == nullptr && conPath == nullptr)) {
        compression = false;
    }
    if (!compression && !decompression) {
        usage();
        exit(0);
    }

    Falcons::Config conf;
    if (r2 == 0) {
        conf = Falcons::Config(r1);
    } else if (r3 == 0) {
        conf = Falcons::Config(r2, r1);
    } else if (r4 == 0) {
        conf = Falcons::Config(r3, r2, r1);
    } else {
        conf = Falcons::Config(r4, r3, r2, r1);
    }
    conf.openmp = enable_openmp;
    if (compression && conPath != nullptr) {
        conf.loadcfg(conPath);
    }

    if (errBoundMode != nullptr) {
        {
            // backward compatible with SZ2
            if (relErrorBound != nullptr) {
                conf.relErrorBound = atof(relErrorBound);
            }
            if (absErrorBound != nullptr) {
                conf.absErrorBound = atof(absErrorBound);
            }
            if (psnrErrorBound != nullptr) {
                conf.psnrErrorBound = atof(psnrErrorBound);
            }
            if (normErrorBound != nullptr) {
                conf.l2normErrorBound = atof(normErrorBound);
            }
        }

        Falcons::match_enum(errBoundMode, Falcons::EB_MAP, conf.errorBoundMode);
        if (strcmp(errBoundMode, "VR_REL") == 0) {
            conf.errorBoundMode = Falcons::EB_REL;
        }
        if (conf.errorBoundMode == Falcons::EB_ABS) {
            if (errBound != nullptr) {
                conf.absErrorBound = atof(errBound);
            }
        } else if (conf.errorBoundMode == Falcons::EB_REL) {
            if (errBound != nullptr) {
                conf.relErrorBound = atof(errBound);
            }
        } else if (conf.errorBoundMode == Falcons::EB_PSNR) {
            if (errBound != nullptr) {
                conf.psnrErrorBound = atof(errBound);
            }
        } else if (conf.errorBoundMode == Falcons::EB_L2NORM) {
            if (errBound != nullptr) {
                conf.l2normErrorBound = atof(errBound);
            }
        } else if (conf.errorBoundMode == Falcons::EB_ABS_AND_REL) {
        } else if (conf.errorBoundMode == Falcons::EB_ABS_OR_REL) {
        } else {
            printf("Error: wrong error bound mode setting by using the option '-M'\n");
            usage();
            exit(0);
        }
    }

    if (compression) {
        if (dataType == Falcons_FLOAT) {
            compress<float>(inPath, cmpPath, conf);
// #if (!Falcons_DEBUG_TIMINGS)
        } else if (dataType == Falcons_DOUBLE) {
            compress<double>(inPath, cmpPath, conf);
        } else if (dataType == Falcons_UINT16) {
            compress<uint16_t>(inPath, cmpPath, conf);
        } else if (dataType == Falcons_INT16) {
            compress<int16_t>(inPath, cmpPath, conf);
        } else if (dataType == Falcons_UINT32) {
            compress<uint32_t>(inPath, cmpPath, conf);
        } else if (dataType == Falcons_INT32) {
            compress<int32_t>(inPath, cmpPath, conf);
        } else if (dataType == Falcons_UINT64) {
            compress<uint64_t>(inPath, cmpPath, conf);
        } else if (dataType == Falcons_INT64) {
            compress<int64_t>(inPath, cmpPath, conf);
// #endif
        } else {
            printf("Error: data type not supported \n");
            usage();
            exit(0);
        }
    }
    if (decompression) {
        if (printCmpResults && inPath == nullptr) {
            printf("Error: Since you add -a option (analysis), please specify the original data path by -i <path>.\n");
            exit(0);
        }

        if (dataType == Falcons_FLOAT) {

            decompress<float>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
//#if (!Falcons_DEBUG_TIMINGS)
        } else if (dataType == Falcons_DOUBLE) {
            decompress<double>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
        } else if (dataType == Falcons_UINT16) {
            decompress<uint16_t>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
        } else if (dataType == Falcons_INT16) {
            decompress<int16_t>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
        } else if (dataType == Falcons_UINT32) {
            decompress<uint32_t>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
        } else if (dataType == Falcons_INT32) {
            decompress<int32_t>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
        } else if (dataType == Falcons_UINT64) {
            decompress<uint64_t>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
        } else if (dataType == Falcons_INT64) {
            decompress<int64_t>(inPath, cmpPath, decPath, conf, binaryOutput, printCmpResults);
//#endif
        } else {
            printf("Error: data type not supported \n");
            usage();
            exit(0);
        }
    }
    if (printMeta) {
        conf.print();
    }
    if (delCmpPath) {
        remove(cmpPath);
    }
    return 0;
}
