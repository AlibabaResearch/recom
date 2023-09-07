// Copyright 2023 The RECom Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <absl/strings/str_join.h>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stack>
#include <string>
#include <symengine/number.h>
#include <symengine/printers.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tuple>
#include <vector>

#include "cuda_emitter.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

std::string CudaEmitter::MakeTensorVarName(const std::string &tensor_name) {
  NodeDef *node = node_mapping.at(GetNodeNameByTensor(tensor_name));
  int idx = GetOutputIdxByTensor(tensor_name);
  return MakeTensorVarName(node, idx);
}

std::string CudaEmitter::MakeTensorVarName(const NodeDef *node, int idx) {
  const int &node_id = node_id_mapping.at(node->name());
  const std::string &var_name =
      "node_" + std::to_string(node_id) + "_idx_" + std::to_string(idx);
  RECOM_VLOG << "node " << node->name() << ", op " << node->op() << ", var "
             << var_name;
  return var_name;
}

std::string
CudaEmitter::MakeTensorShapeVarName(const std::string &tensor_name) {
  return MakeTensorVarName(tensor_name) + "_shape";
}

std::string CudaEmitter::MakeTensorShapeVarName(const NodeDef *node, int idx) {
  return MakeTensorVarName(node, idx) + "_shape";
}

bool CudaEmitter::IsReshape(const NodeDef *node, const int out_idx,
                            int &in_idx) {
  if (node->op() == "Reshape" || node->op() == "ExpandDims" ||
      node->op() == "Squeeze") {
    RETURN_IF_FALSE(out_idx == 0);
    in_idx = 0;
    return true;
  } else {
    // TODO: add more cases
    return false;
  }
}

bool CudaEmitter::IsReshape(const NodeDef *node, const int out_idx) {
  int in_idx;
  return IsReshape(node, out_idx, in_idx);
}

bool CudaEmitter::Optimize() {
  std::vector<std::shared_ptr<FCMeta>> fc_metas;
  std::string code;
  std::vector<bool> success_flags;
  RETURN_IF_FALSE(EmitCodes(fc_metas, code, success_flags));

  const std::string &code_md5 = GetStringMD5(code);
  RETURN_IF_FALSE(code_md5.size() == 32);

  bool debug_mode = GetEnv("RECOM_DEBUG", "off") == "on";
  const std::string &dlpath =
      output_dir + "/" + code_md5 + (debug_mode ? "_debug" : "") + ".so";
  if (!ExistFile(dlpath)) {
    const std::string &code_path = output_dir + "/" + code_md5 + ".cu";
    std::ofstream code_file(code_path);
    code_file << code << std::endl;

    if (GetEnv("RECOM_FORMAT_GEN_CODES", "off") == "on") {
      if (system(("clang-format -i " + code_path).c_str()) != 0) {
        LOG(WARNING) << "Fail to format the codes. Please ensure clang-format "
                        "is installed.";
      }
    }
    LOG(INFO) << "Start compiling " << code_path;

    std::string compile_cmd = "nvcc -Xcompiler -fPIC --shared -o " + dlpath +
                              " " + code_path +
                              (debug_mode ? " -O0 -G -g" : " -O3");
    RETURN_IF_FALSE(system(compile_cmd.c_str()) == 0);
  } else {
    LOG(INFO) << dlpath << " already exists in the cache!";
  }

  RETURN_IF_FALSE(Rewrite(fc_metas, dlpath));

  return true;
}

bool CudaEmitter::EmitCodes(std::vector<std::shared_ptr<FCMeta>> &fc_metas,
                            std::string &code,
                            std::vector<bool> &success_flags) {
  RETURN_IF_FALSE(EmitHeaders(code));
  code += "\n";

  const int num_fc = fc_node_sets.size();
  success_flags = std::vector<bool>(num_fc, false);
  for (int fc_id = 0, emit_id = 0; fc_id < num_fc; ++fc_id) {
    auto fc_meta_ptr = std::make_shared<FCMeta>();
    std::string fc_code_string;
    if (EmitFCCode(fc_node_sets[fc_id], fc_boundary_node_sets[fc_id], emit_id,
                   *fc_meta_ptr, fc_code_string)) {
      fc_metas.push_back(fc_meta_ptr);
      code += "\n" + fc_code_string;
      success_flags[fc_id] = true;
      ++emit_id;

      RECOM_VLOG << "Emit FC " << fc_id << " successfully";
    }
  }
  code += "\n";

  for (int fc_id = 0; fc_id < num_fc; ++fc_id) {
    if (!success_flags[fc_id]) {
      RETURN_IF_FALSE(
          SetUnemitFCToCPU(fc_node_sets[fc_id], fc_boundary_node_sets[fc_id]));
    }
  }

  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &tensor_tuple : fc_meta_ptr->host_input_tensor_tuples) {
      const std::string &tensor_name = std::get<0>(tensor_tuple);
      NodeDef *input_node = node_mapping.at(GetNodeNameByTensor(tensor_name));
      RETURN_IF_FALSE(SetFCBeginToCPU(input_node));
    }
  }

  std::string kernel;
  RETURN_IF_FALSE(ConstructKernelEntry(fc_metas, kernel));
  code += kernel + "\n";

  std::string const_buff_create;
  RETURN_IF_FALSE(
      ConstructConstBufferPrepareEntry(fc_metas, const_buff_create));
  code += const_buff_create + "\n";

  std::string entry;
  RETURN_IF_FALSE(ConstructOpComputeEntry(fc_metas, entry));
  code += entry + "\n";

  return true;
}

bool CudaEmitter::SetFCBeginToCPU(NodeDef *kernel_inode) {
  std::stack<NodeDef *> node_stack;
  node_stack.push(kernel_inode);
  HashSetT<NodeDef *> visited;
  while (!node_stack.empty()) {
    NodeDef *node = node_stack.top();
    node_stack.pop();

    if (!visited.count(node)) {
      node->set_device("/device:CPU:0");
      for (int i = 0; i < node->input_size(); ++i) {
        node_stack.push(GetInputNode(node, i));
      }
      visited.insert(node);
    }
  }

  return true;
}

bool CudaEmitter::SetUnemitFCToCPU(
    const HashSetT<std::string> &fc_node_set,
    const HashSetT<std::string> &fc_boundary_node_set) {
  for (const std::string &node_name : fc_node_set) {
    NodeDef *node = node_mapping.at(node_name);
    node->set_device("/device:CPU:0");
  }

  for (const std::string &boundary_name : fc_boundary_node_set) {
    for (const std::string &out_name : out_mapping.at(boundary_name)) {
      NodeDef *out_node = node_mapping.at(out_name);
      // if (out_node->op() == "ConcatV2") {
      //   out_node->set_device("/device:CPU:0");
      // }
      out_node->set_device("/device:CPU:0");
    }
  }

  return true;
}

bool CudaEmitter::EmitHeaders(std::string &headers) {
  // TODO: use device function pointer to replace the current input-inline
  // implementation (like experiment:SparseSegmentReduce)

  headers = "#include <cstdio>\n"
            "#include <cstdlib>\n"
            "#include <cub/cub.cuh>\n"
            "#include <functional>\n"
            "#include <thread>\n"
            "#include <vector>\n";
  headers += "\n";

  headers += "#define int32 int32_t\n";
  headers += "#define int64 int64_t\n";
  headers += "#define float32 float\n";
  headers += "\n";

  headers += "constexpr int SCAN_DIM = 8;\n";
  headers += "\n";

  headers +=
      "template <int NUM_BOUNDARIES, typename T>\n"
      "__device__ __forceinline__ int Bucketize(\n"
      "    const float (&s_boundaries)[NUM_BOUNDARIES], const T &value) {\n"
      "  int l = 0, r = NUM_BOUNDARIES - 1;\n"
      "  while (l <= r) {\n"
      "    int mid = (l + r) >> 1;\n"
      "    if (value < s_boundaries[mid]) {\n"
      "      r = mid - 1;\n"
      "    } else {\n"
      "      l = mid + 1;\n"
      "    }\n"
      "  }\n"
      "  return r + 1;\n"
      "}\n";
  headers += "\n";

  headers +=
      "template <bool FULL_BLOCK, int EmbedDim, int BLOCK_THREADS, typename "
      "Tparam>\n"
      "__device__ __forceinline__ void GatherRowsToGlbMem(\n"
      "    int (&s_indices)[BLOCK_THREADS], const Tparam *__restrict__ "
      "d_params,\n"
      "    const int indice, Tparam *__restrict__ d_outputs, const int "
      "num_outputs,\n"
      "    const bool execute_flag) {\n"
      "  const int tid = threadIdx.x;\n"
      "\n"
      "  if (FULL_BLOCK || execute_flag) s_indices[tid] = indice;\n"
      "  __syncthreads();\n"
      "\n"
      "#pragma unroll\n"
      "  for (int i = 0; i < EmbedDim; ++i) {\n"
      "    const int output_idx = tid + i * BLOCK_THREADS;\n"
      "    if (FULL_BLOCK || output_idx < num_outputs) {\n"
      "      const int output_row_idx = output_idx / EmbedDim;\n"
      "      const int col_idx = output_idx - output_row_idx * EmbedDim;\n"
      "      const int param_row_idx = s_indices[output_row_idx];\n"
      "      const int param_idx = param_row_idx * EmbedDim + col_idx;\n"
      "      d_outputs[output_idx] = d_params[param_idx];\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int EmbedDim, int BLOCK_THREADS, typename Tparam>\n"
      "__device__ __forceinline__ void GatherRowsToGlbMem(\n"
      "    int (&s_indices)[BLOCK_THREADS], const Tparam *__restrict__ "
      "d_params,\n"
      "    const int indice, Tparam *__restrict__ d_outputs, const int "
      "num_outputs,\n"
      "    const bool full_block, const bool execute_flag) {\n"
      "  if (full_block) {\n"
      "    GatherRowsToGlbMem<true, EmbedDim, BLOCK_THREADS>(\n"
      "        s_indices, d_params, indice, d_outputs, num_outputs, "
      "execute_flag);\n"
      "  } else {\n"
      "    GatherRowsToGlbMem<false, EmbedDim, BLOCK_THREADS>(\n"
      "        s_indices, d_params, indice, d_outputs, num_outputs, "
      "execute_flag);\n"
      "  }\n"
      "}\n";
  headers += "\n";

  headers +=
      "template <bool FULL_BLOCK, int EmbedDim, int BLOCK_THREADS, typename "
      "Tparam>\n"
      "__device__ __forceinline__ void GatherScatterRows(\n"
      "    int (&s_indices)[BLOCK_THREADS], int (&s_row_ids)[BLOCK_THREADS],\n"
      "    const Tparam *__restrict__ d_params, const int indice, const int "
      "row_id,\n"
      "    Tparam *__restrict__ d_outputs, const int num_intermediate,\n"
      "    const bool execute_flag) {\n"
      "  const int tid = threadIdx.x;\n"
      "\n"
      "  if (FULL_BLOCK || execute_flag) {\n"
      "    s_indices[tid] = indice;\n"
      "    s_row_ids[tid] = row_id;\n"
      "  }\n"
      "  __syncthreads();\n"
      "\n"
      "#pragma unroll\n"
      "  for (int i = 0; i < EmbedDim; ++i) {\n"
      "    const int intermediate_idx = tid + i * BLOCK_THREADS;\n"
      "    if (FULL_BLOCK || intermediate_idx < num_intermediate) {\n"
      "      const int intermediate_row_idx = intermediate_idx / EmbedDim;\n"
      "      const int col_idx = intermediate_idx - intermediate_row_idx * "
      "EmbedDim;\n"
      "      const int param_row_idx = s_indices[intermediate_row_idx];\n"
      "      const int param_idx = param_row_idx * EmbedDim + col_idx;\n"
      "      const int output_row_idx = s_row_ids[intermediate_row_idx];\n"
      "      const int output_idx = output_row_idx * EmbedDim + col_idx;\n"
      "      d_outputs[output_idx] = d_params[param_idx];\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int EmbedDim, int BLOCK_THREADS, typename Tparam>\n"
      "__device__ __forceinline__ void GatherScatterRows(\n"
      "    int (&s_indices)[BLOCK_THREADS], int (&s_row_ids)[BLOCK_THREADS],\n"
      "    const Tparam *__restrict__ d_params, const int indice, const int "
      "row_id,\n"
      "    Tparam *__restrict__ d_outputs, const int num_intermediate,\n"
      "    const bool full_block, const bool execute_flag) {\n"
      "  if (full_block) {\n"
      "    GatherScatterRows<true, EmbedDim, BLOCK_THREADS>(\n"
      "        s_indices, s_row_ids, d_params, indice, row_id, d_outputs,\n"
      "        num_intermediate, execute_flag);\n"
      "  } else {\n"
      "    GatherScatterRows<false, EmbedDim, BLOCK_THREADS>(\n"
      "        s_indices, s_row_ids, d_params, indice, row_id, d_outputs,\n"
      "        num_intermediate, execute_flag);\n"
      "  }\n"
      "}\n";
  headers += "\n";

  headers +=
      "template <int ScanDim, typename Tparam> struct ScanVecPair {\n"
      "  Tparam scan_vec[ScanDim];\n"
      "  int scan_key;\n"
      "\n"
      "  __host__ __device__ __forceinline__ ScanVecPair<ScanDim, Tparam>() = "
      "default;\n"
      "\n"
      "  __host__ __device__ __forceinline__\n"
      "  ScanVecPair<ScanDim, Tparam>(const ScanVecPair<ScanDim, Tparam> "
      "&rhv)\n"
      "      : scan_key(rhv.scan_key) {\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      scan_vec[i] = rhv.scan_vec[i];\n"
      "    }\n"
      "  }\n"
      "\n"
      "  __host__ __device__ __forceinline__ ScanVecPair<ScanDim, Tparam> &\n"
      "  operator=(const ScanVecPair<ScanDim, Tparam> &rhv) {\n"
      "    if (&rhv == this) {\n"
      "      return *this;\n"
      "    }\n"
      "    scan_key = rhv.scan_key;\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      scan_vec[i] = rhv.scan_vec[i];\n"
      "    }\n"
      "    return *this;\n"
      "  }\n"
      "};\n"
      "\n"
      "template <int ScanDim, typename Tparam> struct ScanVecScanOp {\n"
      "  __device__ __forceinline__ ScanVecPair<ScanDim, Tparam>\n"
      "  operator()(const ScanVecPair<ScanDim, Tparam> &first,\n"
      "             const ScanVecPair<ScanDim, Tparam> &second) {\n"
      "    ScanVecPair<ScanDim, Tparam> ret;\n"
      "    ret.scan_key = first.scan_key + second.scan_key;\n"
      "    if (second.scan_key) {\n"
      "#pragma unroll\n"
      "      for (int i = 0; i < ScanDim; ++i) {\n"
      "        ret.scan_vec[i] = second.scan_vec[i];\n"
      "      }\n"
      "    } else {\n"
      "#pragma unroll\n"
      "      for (int i = 0; i < ScanDim; ++i) {\n"
      "        ret.scan_vec[i] = first.scan_vec[i] + second.scan_vec[i];\n"
      "      }\n"
      "    }\n"
      "\n"
      "    return ret;\n"
      "  }\n"
      "};\n"
      "\n"
      "template <int ScanDim, int BLOCK_THREADS, typename Tparam>\n"
      "struct SparseSegmentSumTempStorage {\n"
      "  union {\n"
      "    typename cub::BlockScan<ScanVecPair<ScanDim, Tparam>, "
      "BLOCK_THREADS,\n"
      "                            cub::BLOCK_SCAN_WARP_SCANS>::TempStorage "
      "scan;\n"
      "    int row_ids[BLOCK_THREADS + 2];\n"
      "  };\n"
      "  ScanVecPair<ScanDim, Tparam> last_aggregate;\n"
      "};\n"
      "\n"
      "template <int EmbedDim, int ScanDim, int BLOCK_THREADS, typename "
      "Tparam>\n"
      "union SparseSegmentSumTempStorageWrapper {\n"
      "  SparseSegmentSumTempStorage<ScanDim, BLOCK_THREADS, Tparam> normal;\n"
      "  SparseSegmentSumTempStorage<EmbedDim % ScanDim, BLOCK_THREADS, "
      "Tparam> left;\n"
      "};\n"
      "\n"
      "template <bool FULL_BLOCK, int EmbedDim, int ScanDim, int "
      "BLOCK_THREADS,\n"
      "          typename Tparam>\n"
      "__device__ __forceinline__ void\n"
      "SparseSegmentSum(SparseSegmentSumTempStorage<ScanDim, BLOCK_THREADS, "
      "Tparam> &s,\n"
      "                 const Tparam *__restrict__ d_params, const int "
      "indice,\n"
      "                 const int row_id, const int embed_offset,\n"
      "                 Tparam *__restrict__ d_outputs, const int num_inputs,\n"
      "                 const bool execute_flag) {\n"
      "  const int tid = threadIdx.x;\n"
      "\n"
      "  s.row_ids[tid + 1] = row_id;\n"
      "  __syncthreads();\n"
      "\n"
      "  int head_flag = row_id - s.row_ids[tid];\n"
      "  int tail_flag = s.row_ids[tid + 2] - row_id;\n"
      "  __syncthreads();\n"
      "\n"
      "  ScanVecPair<ScanDim, Tparam> scan_vec_pair;\n"
      "  if (FULL_BLOCK || execute_flag) {\n"
      "    scan_vec_pair.scan_key = head_flag;\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      scan_vec_pair.scan_vec[i] =\n"
      "          d_params[indice * EmbedDim + embed_offset + i];\n"
      "    }\n"
      "  }\n"
      "\n"
      "  cub::BlockScan<ScanVecPair<ScanDim, Tparam>, BLOCK_THREADS,\n"
      "                 cub::BLOCK_SCAN_WARP_SCANS>(s.scan)\n"
      "      .InclusiveScan(scan_vec_pair, scan_vec_pair,\n"
      "                     ScanVecScanOp<ScanDim, Tparam>());\n"
      "\n"
      "  if (tail_flag) {\n"
      "    scan_vec_pair =\n"
      "        ScanVecScanOp<ScanDim, Tparam>()(s.last_aggregate, "
      "scan_vec_pair);\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      d_outputs[row_id * EmbedDim + embed_offset + i] =\n"
      "          scan_vec_pair.scan_vec[i];\n"
      "    }\n"
      "  }\n"
      "  __syncthreads();\n"
      "\n"
      "  if (tid + 1 == BLOCK_THREADS) {\n"
      "    if (tail_flag) {\n"
      "      s.last_aggregate = scan_vec_pair;\n"
      "    } else {\n"
      "      s.last_aggregate =\n"
      "          ScanVecScanOp<ScanDim, Tparam>()(s.last_aggregate, "
      "scan_vec_pair);\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int EmbedDim, int ScanDim, int BLOCK_THREADS, typename "
      "Tparam>\n"
      "__device__ __forceinline__ void\n"
      "SparseSegmentSum(SparseSegmentSumTempStorage<ScanDim, BLOCK_THREADS, "
      "Tparam> &s,\n"
      "                 const Tparam *__restrict__ d_params, const int "
      "indice,\n"
      "                 const int row_id, const int embed_offset,\n"
      "                 Tparam *__restrict__ d_outputs, const int num_inputs,\n"
      "                 const bool full_block, const bool execute_flag) {\n"
      "  if (full_block) {\n"
      "    SparseSegmentSum<true, EmbedDim, ScanDim, BLOCK_THREADS, Tparam>(\n"
      "        s, d_params, indice, row_id, embed_offset, d_outputs, "
      "num_inputs,\n"
      "        execute_flag);\n"
      "  } else {\n"
      "    SparseSegmentSum<false, EmbedDim, ScanDim, BLOCK_THREADS, Tparam>(\n"
      "        s, d_params, indice, row_id, embed_offset, d_outputs, "
      "num_inputs,\n"
      "        execute_flag);\n"
      "  }\n"
      "}\n";
  headers += "\n";

  headers +=
      "template <int ScanDim, typename Tparam> struct ScanVecCntTuple {\n"
      "  Tparam scan_vec[ScanDim];\n"
      "  int counter;\n"
      "  int scan_key;\n"
      "\n"
      "  __host__ __device__ __forceinline__ ScanVecCntTuple<ScanDim, "
      "Tparam>() =\n"
      "      default;\n"
      "\n"
      "  __host__ __device__ __forceinline__\n"
      "  ScanVecCntTuple<ScanDim, Tparam>(const ScanVecCntTuple<ScanDim, "
      "Tparam> &rhv)\n"
      "      : scan_key(rhv.scan_key), counter(rhv.counter) {\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      scan_vec[i] = rhv.scan_vec[i];\n"
      "    }\n"
      "  }\n"
      "\n"
      "  __host__ __device__ __forceinline__ ScanVecCntTuple<ScanDim, Tparam> "
      "&\n"
      "  operator=(const ScanVecCntTuple<ScanDim, Tparam> &rhv) {\n"
      "    if (&rhv == this) {\n"
      "      return *this;\n"
      "    }\n"
      "    scan_key = rhv.scan_key;\n"
      "    counter = rhv.counter;\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      scan_vec[i] = rhv.scan_vec[i];\n"
      "    }\n"
      "    return *this;\n"
      "  }\n"
      "};\n"
      "\n"
      "template <int ScanDim, typename Tparam> struct ScanVecCntScanOp {\n"
      "  __device__ __forceinline__ ScanVecCntTuple<ScanDim, Tparam>\n"
      "  operator()(const ScanVecCntTuple<ScanDim, Tparam> &first,\n"
      "             const ScanVecCntTuple<ScanDim, Tparam> &second) {\n"
      "    ScanVecCntTuple<ScanDim, Tparam> ret;\n"
      "    ret.scan_key = first.scan_key + second.scan_key;\n"
      "    if (second.scan_key) {\n"
      "#pragma unroll\n"
      "      for (int i = 0; i < ScanDim; ++i) {\n"
      "        ret.scan_vec[i] = second.scan_vec[i];\n"
      "      }\n"
      "      ret.counter = second.counter;\n"
      "    } else {\n"
      "#pragma unroll\n"
      "      for (int i = 0; i < ScanDim; ++i) {\n"
      "        ret.scan_vec[i] = first.scan_vec[i] + second.scan_vec[i];\n"
      "      }\n"
      "      ret.counter = first.counter + second.counter;\n"
      "    }\n"
      "\n"
      "    return ret;\n"
      "  }\n"
      "};\n"
      "\n"
      "template <int ScanDim, int BLOCK_THREADS, typename Tparam>\n"
      "struct SparseSegmentMeanTempStorage {\n"
      "  union {\n"
      "    typename cub::BlockScan<ScanVecCntTuple<ScanDim, Tparam>, "
      "BLOCK_THREADS,\n"
      "                            cub::BLOCK_SCAN_WARP_SCANS>::TempStorage "
      "scan;\n"
      "    int row_ids[BLOCK_THREADS + 2];\n"
      "  };\n"
      "  ScanVecCntTuple<ScanDim, Tparam> last_aggregate;\n"
      "};\n"
      "\n"
      "template <int EmbedDim, int ScanDim, int BLOCK_THREADS, typename "
      "Tparam>\n"
      "union SparseSegmentMeanTempStorageWrapper {\n"
      "  SparseSegmentMeanTempStorage<ScanDim, BLOCK_THREADS, Tparam> normal;\n"
      "  SparseSegmentMeanTempStorage<EmbedDim % ScanDim, BLOCK_THREADS, "
      "Tparam> left;\n"
      "};\n"
      "\n"
      "template <bool FULL_BLOCK, int EmbedDim, int ScanDim, int "
      "BLOCK_THREADS,\n"
      "          typename Tparam>\n"
      "__device__ __forceinline__ void SparseSegmentMean(\n"
      "    SparseSegmentMeanTempStorage<ScanDim, BLOCK_THREADS, Tparam> &s,\n"
      "    const Tparam *__restrict__ d_params, const int indice, const int "
      "row_id,\n"
      "    const int embed_offset, Tparam *__restrict__ d_outputs, const int "
      "num_input,\n"
      "    const bool execute_flag) {\n"
      "  const int tid = threadIdx.x;\n"
      "\n"
      "  s.row_ids[tid + 1] = row_id;\n"
      "  __syncthreads();\n"
      "\n"
      "  int head_flag = row_id - s.row_ids[tid];\n"
      "  int tail_flag = s.row_ids[tid + 2] - row_id;\n"
      "  __syncthreads();\n"
      "\n"
      "  ScanVecCntTuple<ScanDim, Tparam> scan_vec_cnt_tuple;\n"
      "  if (FULL_BLOCK || execute_flag) {\n"
      "    scan_vec_cnt_tuple.scan_key = head_flag;\n"
      "    scan_vec_cnt_tuple.counter = 1;\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      scan_vec_cnt_tuple.scan_vec[i] =\n"
      "          d_params[indice * EmbedDim + embed_offset + i];\n"
      "    }\n"
      "  }\n"
      "\n"
      "  cub::BlockScan<ScanVecCntTuple<ScanDim, Tparam>, BLOCK_THREADS,\n"
      "                 cub::BLOCK_SCAN_WARP_SCANS>(s.scan)\n"
      "      .InclusiveScan(scan_vec_cnt_tuple, scan_vec_cnt_tuple,\n"
      "                     ScanVecCntScanOp<ScanDim, Tparam>());\n"
      "\n"
      "  if (tail_flag) {\n"
      "    scan_vec_cnt_tuple = ScanVecCntScanOp<ScanDim, Tparam>()(\n"
      "        s.last_aggregate, scan_vec_cnt_tuple);\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < ScanDim; ++i) {\n"
      "      d_outputs[row_id * EmbedDim + embed_offset + i] =\n"
      "          scan_vec_cnt_tuple.scan_vec[i] / scan_vec_cnt_tuple.counter;\n"
      "    }\n"
      "  }\n"
      "  __syncthreads();\n"
      "\n"
      "  if (tid + 1 == BLOCK_THREADS) {\n"
      "    if (tail_flag) {\n"
      "      s.last_aggregate = scan_vec_cnt_tuple;\n"
      "    } else {\n"
      "      s.last_aggregate = ScanVecCntScanOp<ScanDim, Tparam>()(\n"
      "          s.last_aggregate, scan_vec_cnt_tuple);\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int EmbedDim, int ScanDim, int BLOCK_THREADS, typename "
      "Tparam>\n"
      "__device__ __forceinline__ void SparseSegmentMean(\n"
      "    SparseSegmentMeanTempStorage<ScanDim, BLOCK_THREADS, Tparam> &s,\n"
      "    const Tparam *__restrict__ d_params, const int indice, const int "
      "row_id,\n"
      "    const int embed_offset, Tparam *__restrict__ d_outputs, const int "
      "num_input,\n"
      "    const bool full_block, const bool execute_flag) {\n"
      "  if (full_block) {\n"
      "    SparseSegmentMean<true, EmbedDim, ScanDim, BLOCK_THREADS, Tparam>(\n"
      "        s, d_params, indice, row_id, embed_offset, d_outputs, "
      "num_input,\n"
      "        execute_flag);\n"
      "  } else {\n"
      "    SparseSegmentMean<false, EmbedDim, ScanDim, BLOCK_THREADS, "
      "Tparam>(\n"
      "        s, d_params, indice, row_id, embed_offset, d_outputs, "
      "num_input,\n"
      "        execute_flag);\n"
      "  }\n"
      "}\n";
  headers += "\n";

  headers +=
      "// Ported from TensorFlow 2.6\n"
      "// Represents an aligned array of N elements of T. Data pointers can "
      "be\n"
      "// reinterpreted as this type to generate vectorized loads/stores in a "
      "kernel.\n"
      "template <typename T, int N> class alignas(alignof(T) * N) "
      "AlignedVector {\n"
      "public:\n"
      "  typedef T value_type;\n"
      "  static constexpr const int kSize = N;\n"
      "\n"
      "  AlignedVector() = default;\n"
      "\n"
      "  // Uniform initialization.\n"
      "  __host__ __device__ explicit AlignedVector(value_type uniform) {\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < kSize; ++i) {\n"
      "      values_[i] = uniform;\n"
      "    }\n"
      "  }\n"
      "  // Uniform initialization with explicit conversion.\n"
      "  // Note: This is required for T=Eigen::half because it only supports "
      "explicit\n"
      "  // conversions from other types and its template constructor is too "
      "relaxed\n"
      "  // to be able to use std::is_constructible.\n"
      "  template <typename U, typename "
      "std::enable_if<std::is_arithmetic<U>::value,\n"
      "                                                int>::type = 0>\n"
      "  __host__ __device__ explicit AlignedVector(U uniform_u) {\n"
      "    value_type uniform(uniform_u);\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < kSize; ++i) {\n"
      "      values_[i] = uniform;\n"
      "    }\n"
      "  }\n"
      "  // Implicit conversion.\n"
      "  template <typename U, typename std::enable_if<\n"
      "                            std::is_convertible<U, T>::value, "
      "int>::type = 0>\n"
      "  __host__ __device__ AlignedVector(const AlignedVector<U, N> &other) "
      "{\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < kSize; ++i) {\n"
      "      values_[i] = other[i];\n"
      "    }\n"
      "  }\n"
      "  // Explicit conversion.\n"
      "  template <typename U,\n"
      "            typename std::enable_if<!std::is_convertible<U, T>::value "
      "&&\n"
      "                                        std::is_constructible<T, "
      "U>::value,\n"
      "                                    int>::type = 0>\n"
      "  __host__ __device__ explicit AlignedVector(const AlignedVector<U, N> "
      "&other) {\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < kSize; ++i) {\n"
      "      values_[i] = T(other[i]);\n"
      "    }\n"
      "  }\n"
      "\n"
      "  __host__ __device__ value_type &operator[](int i) { return "
      "values_[i]; }\n"
      "  __host__ __device__ const value_type &operator[](int i) const {\n"
      "    return values_[i];\n"
      "  }\n"
      "\n"
      "#define DEFINE_BINARY_UPDATE_OPERATOR(op) \\\n"
      "  __host__ __device__ AlignedVector &operator op(const AlignedVector "
      "&rhs) { \\\n"
      "    _Pragma(\"unroll\") for (int i = 0; i < kSize; ++i) { \\\n"
      "      values_[i] op rhs[i]; \\\n"
      "    } \\\n"
      "    return *this; \\\n"
      "  }\n"
      "  DEFINE_BINARY_UPDATE_OPERATOR(+=)\n"
      "  DEFINE_BINARY_UPDATE_OPERATOR(-=)\n"
      "  DEFINE_BINARY_UPDATE_OPERATOR(*=)\n"
      "  DEFINE_BINARY_UPDATE_OPERATOR(/=)\n"
      "#undef DEFINE_BINARY_UPDATE_OPERATOR\n"
      "\n"
      "#define DEFINE_BINARY_OPERATOR(op) \\\n"
      "  friend __host__ __device__ AlignedVector operator op( \\\n"
      "      const AlignedVector &lhs, const AlignedVector &rhs) { \\\n"
      "    AlignedVector ret; \\\n"
      "    _Pragma(\"unroll\") for (int i = 0; i < kSize; ++i) { \\\n"
      "      ret[i] = lhs[i] op rhs[i]; \\\n"
      "    } \\\n"
      "    return ret; \\\n"
      "  }\n"
      "  DEFINE_BINARY_OPERATOR(+)\n"
      "  DEFINE_BINARY_OPERATOR(-)\n"
      "  DEFINE_BINARY_OPERATOR(*)\n"
      "  DEFINE_BINARY_OPERATOR(/)\n"
      "#undef DEFINE_BINARY_OPERATOR\n"
      "\n"
      "private:\n"
      "  value_type values_[N];\n"
      "};\n"
      "\n"
      "namespace experiment {\n"
      "\n"
      "template <int ITEMS_PER_THREAD, int BLOCK_THREADS, bool FIRST_BLOCK,\n"
      "          typename AccessSegmentIdT>\n"
      "__device__ __forceinline__ void\n"
      "ComputeSegmentOffsets(const AccessSegmentIdT &access_segment_ids,\n"
      "                      int *__restrict__ d_segment_offsets, const int "
      "num_inputs,\n"
      "                      const int num_segments, const int bid) {\n"
      "  constexpr int TILE_SIZE = ITEMS_PER_THREAD * BLOCK_THREADS;\n"
      "  const int tid = threadIdx.x;\n"
      "  const int tile_offset = bid * TILE_SIZE;\n"
      "\n"
      "  int segment_id[ITEMS_PER_THREAD + 1];\n"
      "#pragma unroll\n"
      "  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD + 1; ++ITEM) {\n"
      "    const int idx = tile_offset + tid * ITEMS_PER_THREAD + ITEM - 1;\n"
      "    segment_id[ITEM] =\n"
      "        idx < num_inputs\n"
      "            ? (FIRST_BLOCK && idx < 0 ? -1 : access_segment_ids(idx))\n"
      "            : num_segments;\n"
      "  }\n"
      "\n"
      "#pragma unroll\n"
      "  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {\n"
      "    const int idx = tile_offset + tid * ITEMS_PER_THREAD + ITEM;\n"
      "    for (int id = segment_id[ITEM] + 1; id <= segment_id[ITEM + 1]; "
      "++id) {\n"
      "      d_segment_offsets[id] = idx;\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int ITEMS_PER_THREAD, int BLOCK_THREADS, typename "
      "AccessSegmentIdT>\n"
      "__device__ __forceinline__ void\n"
      "ComputeSegmentOffsets(const AccessSegmentIdT &access_segment_ids,\n"
      "                      int *__restrict__ d_segment_offsets, const int "
      "num_inputs,\n"
      "                      const int num_segments) {\n"
      "  const int num_blocks =\n"
      "      (num_inputs - 1 + 1) / (ITEMS_PER_THREAD * BLOCK_THREADS) + 1;\n"
      "  if (num_blocks > 0) {\n"
      "    ComputeSegmentOffsets<ITEMS_PER_THREAD, BLOCK_THREADS, true>(\n"
      "        access_segment_ids, d_segment_offsets, num_inputs, "
      "num_segments, 0);\n"
      "    for (int bid = 1; bid < num_blocks; ++bid) {\n"
      "      ComputeSegmentOffsets<ITEMS_PER_THREAD, BLOCK_THREADS, false>(\n"
      "          access_segment_ids, d_segment_offsets, num_inputs, "
      "num_segments, bid);\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <typename Tparam, int VECTOR_SIZE, int VBLOCK_DIM_X, int "
      "VBLOCK_DIM_Y>\n"
      "struct SparseSegmentReduceTempStorage {\n"
      "  // AlignedVector<Tparam, VECTOR_SIZE>\n"
      "  //     partial_reduction[VBLOCK_DIM_Y]\n"
      "  //                      [VBLOCK_DIM_X & 1 ? VBLOCK_DIM_X : "
      "VBLOCK_DIM_X + 1];\n"
      "  AlignedVector<Tparam, VECTOR_SIZE> partial_reduction[VBLOCK_DIM_Y]\n"
      "                                                      [VBLOCK_DIM_X];\n"
      "};\n"
      "\n"
      "template <int EMBED_DIM, int VECTOR_SIZE, int SEGMENTS_PER_BLOCK,\n"
      "          int VBLOCK_DIM_X, int VBLOCK_DIM_Y, int "
      "SEQUENCE_UNROLL_FACTOR,\n"
      "          bool MEAN, typename Tparam, typename AccessIndicesT>\n"
      "__device__ __forceinline__ void SparseSegmentReduce(\n"
      "    SparseSegmentReduceTempStorage<Tparam, VECTOR_SIZE, VBLOCK_DIM_X,\n"
      "                                   VBLOCK_DIM_Y> &temp_storage,\n"
      "    const Tparam *__restrict__ d_params, const AccessIndicesT "
      "&access_indices,\n"
      "    const int *__restrict__ d_segment_offsets, Tparam *__restrict__ "
      "d_outputs,\n"
      "    const int num_inputs, const int num_segments, const int bid) {\n"
      "  const int tid = threadIdx.x;\n"
      "  const int ty = tid / VBLOCK_DIM_X;\n"
      "  const int tx = tid - ty * VBLOCK_DIM_X;\n"
      "\n"
      "  static_assert(EMBED_DIM % VECTOR_SIZE == 0,\n"
      "                \"TODO: handle EMBED_DIM not divisible by "
      "VECTOR_SIZE\");\n"
      "  constexpr int VECTORIZED_EMBED_DIM = EMBED_DIM / VECTOR_SIZE;\n"
      "  using Tvec = AlignedVector<Tparam, VECTOR_SIZE>;\n"
      "  auto *__restrict__ d_vectorized_params =\n"
      "      reinterpret_cast<const Tvec *>(d_params);\n"
      "  auto *__restrict__ d_vectorized_outputs = reinterpret_cast<Tvec "
      "*>(d_outputs);\n"
      "\n"
      "  // #pragma unroll\n"
      "  for (int x_offset = 0; x_offset < VECTORIZED_EMBED_DIM;\n"
      "       x_offset += VBLOCK_DIM_X) {\n"
      "    const int x = x_offset + tx;\n"
      "    const bool x_ok = x < VECTORIZED_EMBED_DIM;\n"
      "#pragma unroll\n"
      "    for (int ITEM = 0; ITEM < SEGMENTS_PER_BLOCK; ++ITEM) {\n"
      "      const int segment_id = bid * SEGMENTS_PER_BLOCK + ITEM;\n"
      "      const int begin = d_segment_offsets[segment_id];\n"
      "      const int end = d_segment_offsets[segment_id + 1];\n"
      "\n"
      "      Tvec thread_data(0);\n"
      "#pragma unroll SEQUENCE_UNROLL_FACTOR\n"
      "      for (int segment_offset = begin; segment_offset < end;\n"
      "           segment_offset += VBLOCK_DIM_Y) {\n"
      "        const int y_idx = segment_offset + ty;\n"
      "        const bool y_ok = y_idx < end;\n"
      "\n"
      "        if (x_ok && y_ok) {\n"
      "          const int lookup_idx = access_indices(y_idx);\n"
      "          // separate load and add to enable memory coalescing\n"
      "          auto params =\n"
      "              d_vectorized_params[lookup_idx * VECTORIZED_EMBED_DIM + "
      "x];\n"
      "          thread_data += params;\n"
      "        }\n"
      "      }\n"
      "\n"
      "#pragma unroll\n"
      "      for (int stride = (VBLOCK_DIM_Y + 1) / 2, last_stride = "
      "VBLOCK_DIM_Y;\n"
      "           last_stride != stride; stride = ((last_stride = stride) + 1) "
      "/ 2) {\n"
      "        if (x_ok && ty < last_stride) {\n"
      "          temp_storage.partial_reduction[ty][tx] = thread_data;\n"
      "        }\n"
      "        __syncthreads();\n"
      "        if (x_ok && ty + stride < last_stride) {\n"
      "          thread_data += temp_storage.partial_reduction[ty + "
      "stride][tx];\n"
      "        }\n"
      "        __syncthreads();\n"
      "      }\n"
      "\n"
      "      if (ty == 0 && x_ok) {\n"
      "        d_vectorized_outputs[segment_id * VECTORIZED_EMBED_DIM + x] =\n"
      "            MEAN ? thread_data / Tvec(static_cast<Tparam>(end - "
      "begin))\n"
      "                 : thread_data;\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int EMBED_DIM, int VECTOR_SIZE, int SEGMENTS_PER_BLOCK,\n"
      "          int VBLOCK_DIM_X, int VBLOCK_DIM_Y, int "
      "SEQUENCE_UNROLL_FACTOR,\n"
      "          bool MEAN, typename Tparam, typename AccessIndicesT>\n"
      "__device__ __forceinline__ void SparseSegmentReduce(\n"
      "    SparseSegmentReduceTempStorage<Tparam, VECTOR_SIZE, VBLOCK_DIM_X,\n"
      "                                   VBLOCK_DIM_Y> &temp_storage,\n"
      "    const Tparam *__restrict__ d_params, const AccessIndicesT "
      "&access_indices,\n"
      "    const int *__restrict__ d_segment_offsets, Tparam *__restrict__ "
      "d_outputs,\n"
      "    const int num_inputs, const int num_segments) {\n"
      "  const int num_blocks = (num_segments - 1) / SEGMENTS_PER_BLOCK + 1;\n"
      "  for (int bid = 0; bid < num_blocks; ++bid) {\n"
      "    SparseSegmentReduce<EMBED_DIM, VECTOR_SIZE, SEGMENTS_PER_BLOCK,\n"
      "                        VBLOCK_DIM_X, VBLOCK_DIM_Y, "
      "SEQUENCE_UNROLL_FACTOR,\n"
      "                        MEAN>(temp_storage, d_params, access_indices,\n"
      "                              d_segment_offsets, d_outputs, "
      "num_inputs,\n"
      "                              num_segments, bid);\n"
      "    __syncthreads();\n"
      "  }\n"
      "}\n"
      "\n"
      "template <int EMBED_DIM, int VECTOR_SIZE, int SEGMENT_ID_PER_THREAD,\n"
      "          int SEGMENTS_PER_BLOCK, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,\n"
      "          int SEQUENCE_UNROLL_FACTOR, bool MEAN, typename Tparam,\n"
      "          typename AccessSegmentIdT, typename AccessIndicesT>\n"
      "__device__ void SparseSegmentReduce(\n"
      "    SparseSegmentReduceTempStorage<Tparam, VECTOR_SIZE, VBLOCK_DIM_X,\n"
      "                                   VBLOCK_DIM_Y> &temp_storage,\n"
      "    const AccessSegmentIdT &access_segment_ids,\n"
      "    const AccessIndicesT &access_indices, const Tparam *__restrict__ "
      "d_params,\n"
      "    int *__restrict__ d_segment_offsets, Tparam *__restrict__ "
      "d_outputs,\n"
      "    const int num_inputs, const int num_segments) {\n"
      "  constexpr int BLOCK_THREADS = VBLOCK_DIM_X * VBLOCK_DIM_Y;\n"
      "\n"
      "  ComputeSegmentOffsets<SEGMENT_ID_PER_THREAD, BLOCK_THREADS>(\n"
      "      access_segment_ids, d_segment_offsets, num_inputs, "
      "num_segments);\n"
      "  __syncthreads();\n"
      "\n"
      "  SparseSegmentReduce<EMBED_DIM, VECTOR_SIZE, SEGMENTS_PER_BLOCK, "
      "VBLOCK_DIM_X,\n"
      "                      VBLOCK_DIM_Y, SEQUENCE_UNROLL_FACTOR, MEAN>(\n"
      "      temp_storage, d_params, access_indices, d_segment_offsets, "
      "d_outputs,\n"
      "      num_inputs, num_segments);\n"
      "}\n"
      "\n"
      "} // namespace experiment\n";
  headers += "\n";

  headers += "static __inline__ int alignmem(int x) {\n"
             "  return (x + 127) / 128 * 128;\n"
             "}\n";
  headers += "\n";

  return true;
}

bool CudaEmitter::EmitFCCode(const HashSetT<std::string> &fc_node_set,
                             const HashSetT<std::string> &fc_boundary_node_set,
                             int fc_id, FCMeta &fc_meta,
                             std::string &fc_code_string) {
  RETURN_IF_FALSE(FindFCOutputs(fc_boundary_node_set, fc_meta));

  std::vector<std::shared_ptr<SubgraphCode>> subgraph_codes;
  std::queue<std::pair<NodeDef *, int>> node_queue;
  for (NodeDef *node : fc_meta.value_output_nodes) {
    const int num_tensors = GetOutputTensorNum(node);
    for (int i = 0; i < num_tensors; ++i) {
      node_queue.push({node, i});
    }
  }

  HashSetT<NodeDef *> visited;
  while (!node_queue.empty()) {
    auto &node_port = node_queue.front();
    node_queue.pop();

    if (!visited.count(node_port.first)) {
      auto subgraph_code_ptr = std::make_shared<SubgraphCode>();
      bool is_input;
      RETURN_IF_FALSE(EmitSubgraphCode(node_port.first, node_port.second,
                                       fc_meta, *subgraph_code_ptr, node_queue,
                                       is_input));
      if (!(subgraph_code_ptr->inputs.empty() &&
            subgraph_code_ptr->outputs.empty() &&
            subgraph_code_ptr->shared_params.empty())) {
        subgraph_codes.insert(subgraph_codes.begin(), subgraph_code_ptr);
      }
      if (!is_input)
        visited.insert(node_port.first);
    }
  }

  RETURN_IF_FALSE(subgraph_codes.size() > 0);

  for (const std::string &input_tensor : fc_meta.input_tensors) {
    RETURN_IF_FALSE(symbolic_context->ShapeKnown(input_tensor));
    const ExprVec &input_shape = symbolic_context->GetShape(input_tensor);
    for (const Expression &expr : input_shape) {
      const auto &symbol_node_pairs =
          symbolic_context->RetrieveSymbolExprGenNodePairs(expr);
      for (const auto &symbol_node_pair : symbol_node_pairs) {
        fc_meta.input_symbol_strs.insert(
            SymEngine::str(symbol_node_pair.first));
      }
    }
  }

  for (const Expression &expr : fc_meta.shape_inputs) {
    const auto &symbol_node_pairs =
        symbolic_context->RetrieveSymbolExprGenNodePairs(expr);
    for (const auto &symbol_node_pair : symbol_node_pairs) {
      fc_meta.input_symbol_strs.insert(SymEngine::str(symbol_node_pair.first));
    }
  }

  RETURN_IF_FALSE(
      ConstructFCCode(fc_meta, subgraph_codes, fc_id, fc_code_string));

  return true;
}

bool CudaEmitter::FindFCOutputs(
    const HashSetT<std::string> &fc_boundary_node_set, FCMeta &fc_meta) {
  for (const std::string &boundary_node_name : fc_boundary_node_set) {
    NodeDef *boundary_node = node_mapping.at(boundary_node_name);
    const int num_tensors = GetOutputTensorNum(boundary_node);
    for (int i = 0; i < num_tensors; ++i) {
      const DataType tensor_type = GetOutputType(boundary_node, i);
      const Variable real_output_var(MakeTensorVarName(boundary_node, i),
                                     DataTypeString(tensor_type));
      fc_meta.real_output_vars.push_back(real_output_var);
      fc_meta.real_output_var_mapping[real_output_var.name] = real_output_var;

      const std::string tensor_name = FormTensorName(boundary_node, i);
      RETURN_IF_FALSE(symbolic_context->ShapeKnown(tensor_name));
      const ExprVec shape = symbolic_context->GetShape(tensor_name);
      fc_meta.real_output_var_symbolic_shape_mapping[real_output_var.name] =
          shape;
      fc_meta.output_tensor_tuples.push_back(
          {tensor_name, tensor_type, shape.size()});

      // To handle the last reshape nodes of the FC
      NodeDef *value_node = boundary_node;
      int in_idx, out_idx = i;
      while (IsReshape(value_node, out_idx, in_idx)) {
        out_idx = GetOutputIdxByTensor(value_node->input(in_idx));
        value_node = GetInputNode(value_node, in_idx);
      }
      fc_meta.value_output_nodes.push_back(value_node);
      fc_meta.value_output_real_var_mapping[MakeTensorVarName(
          value_node, out_idx)] = real_output_var;
    }
  }

  return true;
}

bool CudaEmitter::EmitSubgraphCode(
    NodeDef *node, int port, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue, bool &is_input) {
  auto process = [&]() -> bool {
    auto check_table_size = [&](NodeDef *weight, DataType dtype) -> bool {
      if (GetEnv("RECOM_CPU_GPU_CO_RUN", "on") == "off") {
        return true;
      }
      std::vector<int> weight_shape;
      RETURN_IF_FALSE(
          symbolic_context->ShapeStatic(weight->name(), weight_shape));
      uint64 weight_size =
          std::accumulate(weight_shape.begin(), weight_shape.end(), 1,
                          std::multiplies<uint64>()) *
          DataTypeSize(dtype);
      RECOM_VLOG << "[Table Size] " << weight->name() << " " << weight_size
                 << " (" << (weight_size >> 20) << "MB)";
      return weight_size <= max_table_size;
    };

    if (node->op() == "SparseSegmentMean" || node->op() == "SparseSegmentSum" ||
        node->op() == "SparseSegmentMeanWithNumSegments" ||
        node->op() == "SparseSegmentSumWithNumSegments") {
      NodeDef *suspicious_weight = GetInputNode(node, 0);
      if (suspicious_weight->op() == "VariableV2" ||
          suspicious_weight->op() == "Const") {
        RETURN_IF_FALSE(
            check_table_size(suspicious_weight, node->attr().at("T").type()));
        std::vector<int> weight_shape;
        RETURN_IF_FALSE(symbolic_context->ShapeStatic(suspicious_weight->name(),
                                                      weight_shape));
        const int embed_dim = weight_shape.back();
        if (embed_dim <= 20) {
          RETURN_IF_FALSE(
              EmitSparseSegmentReduce(node, fc_meta, code, node_queue));
        } else {
          RETURN_IF_FALSE(EmitSparseSegmentReduceExperiment(node, fc_meta, code,
                                                            node_queue));
        }
      } else {
        LOG_AND_RETURN_FALSE("Currently not support SparseSegmentReduce other "
                             "than Gather Embedding Table");
      }
    } else if (node->op() == "GatherV2") {
      NodeDef *suspicious_weight = GetInputNode(node, 0);
      if (suspicious_weight->op() == "VariableV2" ||
          suspicious_weight->op() == "Const") {
        RETURN_IF_FALSE(check_table_size(suspicious_weight,
                                         node->attr().at("Tparams").type()));
        RETURN_IF_FALSE(EmitGatherRows(node, fc_meta, code, node_queue));
      } else {
        LOG_AND_RETURN_FALSE(
            "Currently not support GatherV2 other than Gather Embedding Table");
      }
    } else if (node->op() == "ScatterNd") {
      auto match_gather_scatter = [&](NodeDef *&weight) -> bool {
        FIND_INPUT_OR_RETURN(node, update, 1, "GatherV2");
        FIND_INPUT_OR_RETURN(update, weight_tmp, 0, "VariableV2,Const");
        weight = weight_tmp;
        return true;
      };

      NodeDef *weight;
      if (match_gather_scatter(weight)) {
        RETURN_IF_FALSE(check_table_size(weight, node->attr().at("T").type()));
        RETURN_IF_FALSE(EmitGatherScatterRows(node, fc_meta, code, node_queue));
      } else {
        LOG_AND_RETURN_FALSE("Currently not support ScatterNd other than "
                             "Gather-Scatter Embedding Table")
      }
    } else if (node->op() == "Sum") {
      // only support BatchColReduction currently
      INPUT_IS_CONST_OR_RETURN(node, 1, 1, int);
      RETURN_IF_FALSE(EmitBatchColReduction(node, fc_meta, code, node_queue));
    } else {
      return false;
    }

    return true;
  };

  if (!process()) {
    RECOM_VLOG << "Unsupported op " << node->op() << " during codegen! "
               << node->name();
    is_input = true;
    const std::string &tensor_name = FormTensorName(node, port);
    if (!fc_meta.input_tensors.count(tensor_name)) {
      const DataType &dtype = GetOutputType(node, port);
      const Variable var(MakeTensorVarName(tensor_name), DataTypeString(dtype));
      RETURN_IF_FALSE(symbolic_context->ShapeKnown(tensor_name));
      const ExprVec &shape_exprs = symbolic_context->GetShape(tensor_name);
      const int &rank = shape_exprs.size();
      const ArrayVariable shape_var(MakeTensorShapeVarName(tensor_name), "int",
                                    rank);
      fc_meta.input_vars.push_back(var);
      fc_meta.host_input_tensor_tuples.push_back({tensor_name, dtype, rank});
      fc_meta.host_input_var_tuples.push_back({var, shape_var, shape_exprs});
      fc_meta.input_tensors.insert(tensor_name);
    }
  }

  return true;
}

bool CudaEmitter::EmitBatchColReduction(
    NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  const NodeDef *sum_node = node;
  const DataType type = sum_node->attr().at("T").type();
  const Variable output_var(MakeTensorVarName(sum_node, 0),
                            DataTypeString(type));
  code.outputs[output_var.name] = output_var;

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(sum_node->name()));
  const ExprVec output_shape = symbolic_context->GetShape(sum_node->name());
  const Expression num_output =
      std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  fc_meta.buffers.push_back({output_var, num_output});

  std::string inline_input;
  const Expression index_expr("item_idx");
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(node, 0),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_input, node_queue));

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(node->input(0)));
  const ExprVec input_shape = symbolic_context->GetShape(node->input(0));
  RETURN_IF_FALSE(input_shape.size() == 3);
  code.pre_glb_area +=
      "    const int batch_size = " + SymEngine::str(input_shape[0]) +
      ";\n"
      "    const int num_rows = " +
      SymEngine::str(input_shape[1]) +
      ";\n"
      "    const int num_cols = " +
      SymEngine::str(input_shape[2]) +
      ";\n"
      "    const int num_output = batch_size * num_cols;\n";

  code.loop_body =
      "    for (int i = 0; i < num_outputs; i += " + block_threads_str +
      ") {\n"
      "    const bool full_block = true;\n"
      "    const bool execute_flag = true;\n"
      "\n"
      "    const int output_idx = i + threadIdx.x;\n"
      "    const int batch_id = output_idx / num_cols;\n"
      "    const int col_idx = output_idx - batch_id * num_cols;\n"
      "    const int input_offset = batch_id * num_rows * num_cols;\n"
      "\n"
      "    if (output_idx < num_outputs) {\n"
      "      " +
      DataTypeString(type) +
      " reduced_val = 0;\n"
      "      for (int j = 0; j < num_rows; ++j) {\n"
      "        const int item_idx = input_offset + j * num_cols;\n"
      "        reduced_val += execute_flag ? " +
      inline_input +
      " : 0;\n"
      "      }\n"
      "      " +
      output_var.At("output_idx") +
      " = reduced_val;\n"
      "    }\n"
      "  }\n";

  return true;
}

bool CudaEmitter::EmitGatherRows(
    NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  const NodeDef *gather_node = node;
  const DataType weight_type = gather_node->attr().at("Tparams").type();
  const Variable output_var(MakeTensorVarName(gather_node, 0),
                            DataTypeString(weight_type));
  code.outputs[output_var.name] = output_var;

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(gather_node->name()));
  const ExprVec output_shape = symbolic_context->GetShape(gather_node->name());
  const Expression num_output =
      std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  fc_meta.buffers.push_back({output_var, num_output});

  const NodeDef *weight_node = GetInputNode(node, 0);
  const Variable weight_var(MakeTensorVarName(node->input(0)),
                            DataTypeString(weight_type));
  const ArrayVariable weight_shape_var(MakeTensorShapeVarName(node->input(0)),
                                       "int", 2);
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(weight_node->name()));
  const ExprVec weight_shape_exprs =
      symbolic_context->GetShape(weight_node->name());
  code.inputs[weight_var.name] = weight_var;
  const std::string &weight_tensor_name = FormTensorName(weight_node, 0);
  if (!fc_meta.input_tensors.count(weight_tensor_name)) {
    fc_meta.input_vars.push_back(weight_var);
    fc_meta.device_input_tensor_tuples.push_back(
        {weight_tensor_name, weight_type, 2});
    fc_meta.device_input_var_tuples.push_back(
        {weight_var, weight_shape_var, weight_shape_exprs});
    fc_meta.input_tensors.insert(weight_tensor_name);
  }

  const ArrayVariable s_indices_var("s_" + output_var.name + "_indices", "int",
                                    block_threads);
  code.shared_params[s_indices_var.name] = s_indices_var;

  std::vector<int> weight_shape;
  RETURN_IF_FALSE(
      symbolic_context->ShapeStatic(weight_node->name(), weight_shape));
  const int embed_dim = weight_shape.back();
  const std::string embed_dim_str = std::to_string(embed_dim);

  std::string inline_input;
  const Expression index_expr("item_idx");
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(node, 1),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_input, node_queue));

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(node->input(1)));
  const ExprVec input_shape = symbolic_context->GetShape(node->input(1));
  code.pre_glb_area += "    const int num_input = 1";
  for (const Expression &expr : input_shape) {
    code.pre_glb_area += " * " + SymEngine::str(expr);
  }
  code.pre_glb_area += ";\n";

  code.loop_body =
      "    for (int i = 0; i < num_input; i += " + block_threads_str +
      ") {\n"
      "      const bool full_block = (i + " +
      block_threads_str +
      ") <= num_input;\n"
      "      const int item_idx = i + threadIdx.x;\n"
      "      const bool execute_flag = item_idx < num_input;\n"
      "      GatherRowsToGlbMem<" +
      embed_dim_str +
      ">(\n"
      "        " +
      s_indices_var.name + ", " + weight_var.name +
      ",\n"
      "        execute_flag ? " +
      inline_input +
      " : 0,\n"
      "        " +
      output_var.name + " + i * " + embed_dim_str + ", (num_input - i) * " +
      embed_dim_str +
      ", full_block, execute_flag);\n"
      "      __syncthreads();\n"
      "    }\n";

  return true;
}

bool CudaEmitter::EmitGatherScatterRows(
    NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  const NodeDef *scatter_node = node;
  const DataType weight_type = scatter_node->attr().at("T").type();
  const Variable output_var(MakeTensorVarName(scatter_node, 0),
                            DataTypeString(weight_type));
  code.outputs[output_var.name] = output_var;

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(scatter_node->name()));
  const ExprVec output_shape = symbolic_context->GetShape(scatter_node->name());
  const Expression num_output =
      std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  fc_meta.buffers.push_back({output_var, num_output});
  // output shape of ScatterNd is determined by its 3rd input
  fc_meta.shape_inputs.insert(fc_meta.shape_inputs.end(), output_shape.begin(),
                              output_shape.end());

  code.pre_glb_area +=
      "    const int num_output = " + SymEngine::str(num_output) + "\n;";
  code.pre_glb_area += "    for (int i = threadIdx.x; i < num_output; i += " +
                       block_threads_str +
                       ") {\n"
                       "      " +
                       output_var.At("i") +
                       " = 0;\n"
                       "}\n";
  code.pre_glb_area += "\n";

  const NodeDef *gather_node = GetInputNode(scatter_node, 1);
  const NodeDef *weight_node = GetInputNode(gather_node, 0);
  const Variable weight_var(MakeTensorVarName(weight_node, 0),
                            DataTypeString(weight_type));
  const ArrayVariable weight_shape_var(MakeTensorShapeVarName(weight_node, 0),
                                       "int", 2);
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(weight_node->name()));
  const ExprVec weight_shape_exprs =
      symbolic_context->GetShape(weight_node->name());
  code.inputs[weight_var.name] = weight_var;
  const std::string &weight_tensor_name = FormTensorName(weight_node, 0);
  if (!fc_meta.input_tensors.count(weight_tensor_name)) {
    fc_meta.input_vars.push_back(weight_var);
    fc_meta.device_input_tensor_tuples.push_back(
        {weight_tensor_name, weight_type, 2});
    fc_meta.device_input_var_tuples.push_back(
        {weight_var, weight_shape_var, weight_shape_exprs});
    fc_meta.input_tensors.insert(weight_tensor_name);
  }

  const ArrayVariable s_indices_var("s_" + output_var.name + "_indices", "int",
                                    block_threads);
  code.shared_params[s_indices_var.name] = s_indices_var;

  const ArrayVariable s_row_ids_var("s_" + output_var.name + "_row_ids", "int",
                                    block_threads);
  code.shared_params[s_row_ids_var.name] = s_row_ids_var;

  std::vector<int> weight_shape;
  RETURN_IF_FALSE(
      symbolic_context->ShapeStatic(weight_node->name(), weight_shape));
  const int embed_dim = weight_shape.back();
  const std::string embed_dim_str = std::to_string(embed_dim);

  std::string inline_indices_input;
  const Expression index_expr("item_idx");
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(gather_node, 1),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_indices_input, node_queue));

  std::string inline_sp_indices_input;
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(scatter_node, 0),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_sp_indices_input, node_queue));

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(gather_node->input(1)));
  const ExprVec input_shape = symbolic_context->GetShape(gather_node->input(1));
  const Expression num_input =
      std::accumulate(input_shape.begin(), input_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  code.pre_glb_area +=
      "    const int num_input = " + SymEngine::str(num_input) + ";\n";

  code.loop_body =
      "    for (int i = 0; i < num_input; i += " + block_threads_str +
      ") {\n"
      "      const bool full_block = (i + " +
      block_threads_str +
      ") <= num_input;\n"
      "      const int item_idx = i + threadIdx.x;\n"
      "      const bool execute_flag = item_idx < num_input;\n"
      "      GatherScatterRows<" +
      embed_dim_str +
      ">(\n"
      "        " +
      s_indices_var.name + ", " + s_row_ids_var.name + ", " + weight_var.name +
      ",\n"
      "        execute_flag ? " +
      inline_indices_input +
      " : 0,\n"
      "        execute_flag ? " +
      inline_sp_indices_input +
      " : 0,\n"
      "        " +
      output_var.name + ", (num_input - i) * " + embed_dim_str +
      ", full_block, execute_flag);\n"
      "      __syncthreads();\n"
      "    }\n";

  return true;
}

bool CudaEmitter::EmitSparseSegmentReduce(
    NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  const NodeDef *lookup_node = node;
  const DataType weight_type = lookup_node->attr().at("T").type();
  const Variable output_var(MakeTensorVarName(lookup_node, 0),
                            DataTypeString(weight_type));
  code.outputs[output_var.name] = output_var;

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(lookup_node->name()));
  const ExprVec output_shape = symbolic_context->GetShape(lookup_node->name());
  const Expression num_output =
      std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  fc_meta.buffers.push_back({output_var, num_output});

  if (lookup_node->op().find("WithNumSegments") != std::string::npos) {
    // output shape of lookup_node is determined by its 4th input
    fc_meta.shape_inputs.insert(fc_meta.shape_inputs.end(),
                                output_shape.begin(), output_shape.end());

    code.pre_glb_area +=
        "    const int num_output = " + SymEngine::str(num_output) + "\n;";
    code.pre_glb_area += "    for (int i = threadIdx.x; i < num_output; i += " +
                         block_threads_str +
                         ") {\n"
                         "      " +
                         output_var.At("i") +
                         " = 0;\n"
                         "}\n";
    code.pre_glb_area += "\n";
  }

  const NodeDef *weight_node = GetInputNode(lookup_node, 0);
  const Variable weight_var(MakeTensorVarName(weight_node, 0),
                            DataTypeString(weight_type));
  const ArrayVariable weight_shape_var(MakeTensorShapeVarName(weight_node, 0),
                                       "int", 2);
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(weight_node->name()));
  const ExprVec weight_shape_exprs =
      symbolic_context->GetShape(weight_node->name());
  code.inputs[weight_var.name] = weight_var;
  const std::string &weight_tensor_name = FormTensorName(weight_node, 0);
  if (!fc_meta.input_tensors.count(weight_tensor_name)) {
    fc_meta.input_vars.push_back(weight_var);
    fc_meta.device_input_tensor_tuples.push_back(
        {weight_tensor_name, weight_type, 2});
    fc_meta.device_input_var_tuples.push_back(
        {weight_var, weight_shape_var, weight_shape_exprs});
    fc_meta.input_tensors.insert(weight_tensor_name);
  }

  std::vector<int> weight_shape;
  RETURN_IF_FALSE(
      symbolic_context->ShapeStatic(weight_node->name(), weight_shape));
  const int embed_dim = weight_shape.back();
  const std::string embed_dim_str = std::to_string(embed_dim);

  const bool combiner_is_mean =
      lookup_node->op().find("SparseSegmentMean") != std::string::npos;
  const ArrayVariable s_var(
      "s_" + output_var.name + "_ssr",
      std::string("SparseSegment") + (combiner_is_mean ? "Mean" : "Sum") +
          "TempStorageWrapper<" + embed_dim_str + ", SCAN_DIM, " +
          block_threads_str + ", " + weight_var.type + ">",
      1);
  code.shared_params[s_var.name] = s_var;

  std::string inline_indices_input;
  const Expression index_expr("item_idx");
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(lookup_node, 1),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_indices_input, node_queue));

  std::string inline_sp_indices_input;
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(lookup_node, 2),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_sp_indices_input, node_queue));

  std::string inline_next_sp_indices_input;
  const Expression next_index_expr = index_expr + 1;
  RETURN_IF_FALSE(EmitInputInline(
      GetInputNode(lookup_node, 2), {next_index_expr, next_index_expr}, 0,
      fc_meta, code, inline_next_sp_indices_input, node_queue));

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(lookup_node->input(1)));
  const ExprVec input_shape = symbolic_context->GetShape(lookup_node->input(1));
  const Expression num_input =
      std::accumulate(input_shape.begin(), input_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  code.pre_glb_area +=
      "    const int num_input = " + SymEngine::str(num_input) + ";\n";
  code.pre_glb_area += "\n";

  code.pre_glb_area +=
      "    constexpr int LEFT_DIM = " + embed_dim_str + " % SCAN_DIM;\n";
  code.pre_glb_area += "\n";

  auto get_process_str = [&](bool is_left) -> std::string {
    const std::string scan_dim_str = is_left ? "LEFT_DIM" : "SCAN_DIM";
    const std::string smem_name =
        s_var.At(0) + "." + (is_left ? "left" : "normal");

    std::string str;
    str += "      for (int i = threadIdx.x; i < " + scan_dim_str +
           "; i += " + block_threads_str +
           ") {\n"
           "        " +
           smem_name +
           ".last_aggregate.scan_vec[i] = 0;\n"
           "      }\n"
           "      if (threadIdx.x == 0) {\n"
           "        " +
           smem_name + ".last_aggregate.scan_key = 0;\n";
    if (combiner_is_mean) {
      str += "        " + smem_name + ".last_aggregate.counter = 0;\n";
    }
    str += "      }\n";
    str += "      __syncthreads();\n";
    str += "\n";

    str += "      int last_row_id = 0;\n"
           "      for (int offset = 0; offset < num_input; offset += " +
           block_threads_str +
           ") {\n"
           "        const bool full_block = (offset + " +
           block_threads_str +
           ") <= num_input;\n"
           "        const int item_idx = offset + threadIdx.x;\n"
           "        const bool execute_flag = item_idx < num_input;\n"
           "        if (threadIdx.x + 1 == " +
           block_threads_str +
           ") {\n"
           "        " +
           smem_name +
           ".row_ids[0] = last_row_id;\n"
           "        " +
           smem_name + ".row_ids[" + block_threads_str +
           " + 1] = (item_idx + 1) < num_input ? " +
           inline_next_sp_indices_input +
           " : INT_MAX;\n"
           "        }\n"
           "        last_row_id = execute_flag ? " +
           inline_sp_indices_input +
           " : INT_MAX;\n"
           "        SparseSegment" +
           (combiner_is_mean ? "Mean" : "Sum") + "<" + embed_dim_str + ", " +
           scan_dim_str + ", " + block_threads_str +
           ", float>(\n"
           "            " +
           smem_name + ", " + weight_var.name +
           ",\n"
           "            execute_flag ? " +
           inline_indices_input +
           " : 0,\n"
           "            last_row_id, embed_offset, " +
           output_var.name +
           ",\n"
           "            num_input - offset, full_block, execute_flag);\n"
           "        __syncthreads();\n"
           "      }\n";

    return str;
  };

  code.loop_body += "    for (int embed_offset = 0; embed_offset < " +
                    embed_dim_str +
                    " - LEFT_DIM; embed_offset += SCAN_DIM) {\n" +
                    get_process_str(false) + "    }\n";
  code.loop_body += "\n";

  code.loop_body += "    if (LEFT_DIM != 0) {\n"
                    "      int embed_offset = " +
                    embed_dim_str + " - LEFT_DIM;\n" + get_process_str(true) +
                    "    }\n";
  code.loop_body += "\n";

  return true;
}

bool CudaEmitter::EmitSparseSegmentReduceExperiment(
    NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  // TODO: avoid hard code
#define DECLARE_HYPER_PARAMETER(param, value)                                  \
  const int param = value;                                                     \
  const std::string param##_str(std::to_string(value));

  DECLARE_HYPER_PARAMETER(vector_size, 4);
  DECLARE_HYPER_PARAMETER(vblock_dim_x, 8);
  DECLARE_HYPER_PARAMETER(vblock_dim_y, block_threads / vblock_dim_x);
  DECLARE_HYPER_PARAMETER(segment_id_per_thread, 1);
  DECLARE_HYPER_PARAMETER(segments_per_block, 1);
  DECLARE_HYPER_PARAMETER(sequence_unroll_factor, 4);

#undef DECLARE_HYPER_PARAMETER

  const NodeDef *lookup_node = node;
  const DataType weight_type = lookup_node->attr().at("T").type();
  const Variable output_var(MakeTensorVarName(lookup_node, 0),
                            DataTypeString(weight_type));
  code.outputs[output_var.name] = output_var;

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(lookup_node->name()));
  const ExprVec output_shape = symbolic_context->GetShape(lookup_node->name());
  const Expression output_elenum =
      std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  fc_meta.buffers.push_back({output_var, output_elenum});

  if (lookup_node->op().find("WithNumSegments") != std::string::npos) {
    // output shape of lookup_node is determined by its 4th input
    fc_meta.shape_inputs.insert(fc_meta.shape_inputs.end(),
                                output_shape.begin(), output_shape.end());

    code.pre_glb_area +=
        "    const int output_elenum = " + SymEngine::str(output_elenum) +
        "\n;";
    code.pre_glb_area +=
        "    for (int i = threadIdx.x; i < output_elenum; i += " +
        block_threads_str +
        ") {\n"
        "      " +
        output_var.At("i") +
        " = 0;\n"
        "}\n";
    code.pre_glb_area += "\n";
  }

  const NodeDef *weight_node = GetInputNode(lookup_node, 0);
  const Variable weight_var(MakeTensorVarName(weight_node, 0),
                            DataTypeString(weight_type));
  const ArrayVariable weight_shape_var(MakeTensorShapeVarName(weight_node, 0),
                                       "int", 2);
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(weight_node->name()));
  const ExprVec weight_shape_exprs =
      symbolic_context->GetShape(weight_node->name());
  code.inputs[weight_var.name] = weight_var;
  const std::string &weight_tensor_name = FormTensorName(weight_node, 0);
  if (!fc_meta.input_tensors.count(weight_tensor_name)) {
    fc_meta.input_vars.push_back(weight_var);
    fc_meta.device_input_tensor_tuples.push_back(
        {weight_tensor_name, weight_type, 2});
    fc_meta.device_input_var_tuples.push_back(
        {weight_var, weight_shape_var, weight_shape_exprs});
    fc_meta.input_tensors.insert(weight_tensor_name);
  }

  std::vector<int> weight_shape;
  RETURN_IF_FALSE(
      symbolic_context->ShapeStatic(weight_node->name(), weight_shape));
  const int embed_dim = weight_shape.back();
  const std::string embed_dim_str = std::to_string(embed_dim);

  const bool combiner_is_mean =
      lookup_node->op().find("SparseSegmentMean") != std::string::npos;
  const ArrayVariable s_var("s_" + output_var.name + "_ssr",
                            "experiment::SparseSegmentReduceTempStorage<" +
                                weight_var.type + ", " + vector_size_str +
                                ", " + vblock_dim_x_str + ", " +
                                vblock_dim_y_str + ">",
                            1);
  code.shared_params[s_var.name] = s_var;

  std::string index_str("idx");
  const Expression index_expr(index_str);

  std::string inline_indices_input;
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(lookup_node, 1),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_indices_input, node_queue));
  const std::string &access_indices_input =
      "[&](int " + index_str + ") { return " + inline_indices_input + ";}";

  std::string inline_sp_indices_input;
  RETURN_IF_FALSE(EmitInputInline(GetInputNode(lookup_node, 2),
                                  {index_expr, index_expr}, 0, fc_meta, code,
                                  inline_sp_indices_input, node_queue));
  const std::string &access_sp_indices_input =
      "[&](int " + index_str + ") { return " + inline_sp_indices_input + ";}";

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(lookup_node->input(1)));
  const ExprVec input_shape = symbolic_context->GetShape(lookup_node->input(1));
  const Expression input_elenum =
      std::accumulate(input_shape.begin(), input_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  code.pre_glb_area +=
      "    const int input_elenum = " + SymEngine::str(input_elenum) + ";\n";
  code.pre_glb_area += "\n";

  const Variable segment_offsets_var(output_var.name + "_segment_offsets",
                                     "int");
  const Expression num_segments = output_elenum / embed_dim;
  fc_meta.buffers.push_back({segment_offsets_var, num_segments + 1});
  code.outputs[segment_offsets_var.name] = segment_offsets_var;

  const std::vector<std::string> &tparams = {embed_dim_str,
                                             vector_size_str,
                                             segment_id_per_thread_str,
                                             segments_per_block_str,
                                             vblock_dim_x_str,
                                             vblock_dim_y_str,
                                             sequence_unroll_factor_str,
                                             combiner_is_mean ? "true"
                                                              : "false"};

  const std::vector<std::string> &params = {
      s_var.At(0),     access_sp_indices_input,     access_indices_input,
      weight_var.name, segment_offsets_var.name,    output_var.name,
      "input_elenum",  SymEngine::str(num_segments)};

  code.loop_body += "    experiment::SparseSegmentReduce<" +
                    absl::StrJoin(tparams, ", ") + ">(\n" +
                    absl::StrJoin(params, ",\n      ") + ");\n";

  return true;
}

bool CudaEmitter::EmitElementWise(
    NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  // TODO
  return false;
}

bool CudaEmitter::EmitInputInline(
    NodeDef *node, const std::pair<Expression, Expression> &index_expr_pair,
    int out_idx, FCMeta &fc_meta, SubgraphCode &code, std::string &inline_code,
    std::queue<std::pair<NodeDef *, int>> &node_queue) {
  std::vector<int> input_indices;
  std::vector<Expression> prev_index_exprs;
  std::function<std::string(const std::vector<std::string> &)>
      process_prev_inlines =
          [=](const std::vector<std::string> &prev_inlines) -> std::string {
    assert(prev_inlines.size() == 1);
    return prev_inlines[0];
  };

  auto process = [&]() -> bool {
    int input_idx;
    if (IsReshape(node, out_idx, input_idx)) {
      // Skip
      prev_index_exprs = {index_expr_pair.first};
      input_indices = {input_idx};
    } else if (node->op() == "Cast") {
      const DataType dst_type = node->attr().at("DstT").type();
      const std::string &dst_type_str = DataTypeString(dst_type);
      process_prev_inlines =
          [=](const std::vector<std::string> &prev_inlines) -> std::string {
        assert(prev_inlines.size() == 1);
        return "static_cast<" + dst_type_str + ">(" + prev_inlines[0] + ")";
      };
      prev_index_exprs = {index_expr_pair.first};
      input_indices = {0};
    } else if (node->op() == "Bucketize") {
      const auto &boundary_list = node->attr().at("boundaries").list();
      const int num_boundaries = boundary_list.f_size();
      const Variable boundaries_var(MakeTensorVarName(node, 0) + "_boundaries",
                                    "float");
      if (!code.inputs.count(boundaries_var.name)) {
        code.inputs[boundaries_var.name] = boundaries_var;

        std::vector<std::string> boundaries_string(num_boundaries);
        for (int i = 0; i < num_boundaries; ++i) {
          boundaries_string[i] = std::to_string(boundary_list.f(i));
        }
        fc_meta.const_buffers.push_back({boundaries_var, boundaries_string});
      }

      const ArrayVariable s_boundaries_var("s_" + boundaries_var.name, "float",
                                           num_boundaries);
      if (!code.shared_params.count(s_boundaries_var.name)) {
        code.shared_params[s_boundaries_var.name] = s_boundaries_var;

        code.pre_glb_area +=
            "    for (int i = threadIdx.x; i < " +
            std::to_string(num_boundaries) + "; i += " + block_threads_str +
            ") {\n"
            "      " +
            s_boundaries_var.At("i") + " = " + boundaries_var.At("i") +
            ";\n"
            "    }\n";
      }

      process_prev_inlines =
          [=](const std::vector<std::string> &prev_inlines) -> std::string {
        assert(prev_inlines.size() == 1);
        return "Bucketize(" + s_boundaries_var.name + ", " + prev_inlines[0] +
               ")";
      };
      prev_index_exprs = {index_expr_pair.first};
      input_indices = {0};
    } else if (node->op() == "StridedSlice") {
      EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(node, 1), begin, int);
      EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(node, 2), end, int);
      EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(node, 3), strides, int);

      auto get_mask = [&](const std::string &key) -> int {
        if (node->attr().contains(key)) {
          return node->attr().at(key).i();
        }
        return 0;
      };

      int ellipsis_mask = get_mask("ellipsis_mask");
      int begin_mask = get_mask("begin_mask");
      int end_mask = get_mask("end_mask");
      int new_axis_mask = get_mask("new_axis_mask");
      int shrink_axis_mask = get_mask("shrink_axis_mask");

      RETURN_IF_FALSE(symbolic_context->ShapeKnown(node->input(0)));
      const ExprVec &input_shape = symbolic_context->GetShape(node->input(0));
      RETURN_IF_FALSE(symbolic_context->ShapeKnown(node->name()));
      const ExprVec &output_shape = symbolic_context->GetShape(node->name());

      RECOM_VLOG << "StridedSlice input_shape: " << ExprVecToStr(input_shape)
                 << ", output_shape: " << ExprVecToStr(output_shape);

      // TODO: support more complicated index transformation
      // Now is hard code for a special case
      RETURN_IF_FALSE(ellipsis_mask == 0 && new_axis_mask == 0 &&
                      shrink_axis_mask == 0);
      RETURN_IF_FALSE(output_shape.size() == 2 && input_shape.size() == 2);
      RETURN_IF_FALSE(begin.size() == 2 && end.size() == 2);
      RETURN_IF_FALSE((begin_mask == 1 || begin[0] == 0) && end_mask == 1);
      RETURN_IF_FALSE(strides[0] == 1);
      RETURN_IF_FALSE(begin[1] == 0 && end[1] == 1);

      prev_index_exprs = {index_expr_pair.first * input_shape[1]};
      input_indices = {0};
    } else if (node->op() == "SparseReshape") {
      RETURN_IF_FALSE(symbolic_context->ContentKnown(node->input(1)));
      const ExprVec &input_shape = symbolic_context->GetContent(node->input(1));
      const int input_rank = input_shape.size();

      RETURN_IF_FALSE(symbolic_context->ContentKnown(node->input(2)));
      const ExprVec &output_shape =
          symbolic_context->GetContent(node->input(2));
      const int output_rank = output_shape.size();

      int offset =
          UnsafeMod(index_expr_pair.first, index_expr_pair.second, output_rank);
      RETURN_IF_FALSE(offset >= 0);

      process_prev_inlines =
          [=](const std::vector<std::string> &prev_inlines) -> std::string {
        assert(prev_inlines.size() == input_dense_rank);
        std::string res = prev_inlines[0];
        for (int i = 1; i < input_rank; ++i) {
          res = "(" + res + ") * (" + SymEngine::str(input_shape[i]) + ") + (" +
                prev_inlines[i] + ")";
        }

        for (int i = output_rank - 1; i > offset; --i) {
          res = "(" + res + ") / (" + SymEngine::str(output_shape[i]) + ")";
        }

        if (offset != 0) {
          res =
              "(" + res + ") % (" + SymEngine::str(output_shape[offset]) + ")";
        }

        return res;
      };

      prev_index_exprs = std::vector<Expression>(input_rank);
      for (int i = 0; i < input_rank; ++i) {
        prev_index_exprs[i] =
            (index_expr_pair.first - offset) / output_rank * input_rank +
            Expression(i);
      }

      input_indices = std::vector<int>(input_rank, 0);
    } else {
      return false;
    }

    return true;
  };

  if (!process()) {
    // TODO: if default value cannot be zero
    RECOM_VLOG << "Subgraph end at " << node->name();
    node_queue.push({node, out_idx});
    const Variable input_var(MakeTensorVarName(node, out_idx),
                             DataTypeString(GetOutputType(node, out_idx)));
    code.inputs[input_var.name] = input_var;
    inline_code = input_var.At(SymEngine::str(index_expr_pair.first));
    return true;
  }

  RETURN_IF_FALSE(prev_index_exprs.size() == input_indices.size());
  std::vector<std::string> prev_inlines(prev_index_exprs.size());
  for (int i = 0; i < prev_inlines.size(); ++i) {
    const int input_idx = input_indices[i];
    NodeDef *prev_node = GetInputNode(node, input_idx);
    int prev_out_idx = GetOutputIdxByTensor(node->input(input_idx));
    RETURN_IF_FALSE(EmitInputInline(
        prev_node, {prev_index_exprs[i], index_expr_pair.second}, prev_out_idx,
        fc_meta, code, prev_inlines[i], node_queue));
  }

  inline_code = process_prev_inlines(prev_inlines);

  return true;
}

bool CudaEmitter::ConstructSubgraphCode(const SubgraphCode &subgraph_code,
                                        int unique_id,
                                        std::string &subgraph_code_string) {
  subgraph_code_string = "  __device__ __forceinline__ void subgraph_code_" +
                         std::to_string(unique_id) + "(\n";

  std::vector<std::string> params;
  for (const auto &input : subgraph_code.inputs) {
    params.push_back("    " + input.second.ConstRestrictPtrParam());
  }
  for (const auto &shared : subgraph_code.shared_params) {
    params.push_back("    " + shared.second.ArrRefParam());
  }
  for (const auto &output : subgraph_code.outputs) {
    params.push_back("    " + output.second.RestrictPtrParam());
  }
  subgraph_code_string += absl::StrJoin(params, ",\n");

  subgraph_code_string += ") {\n" + subgraph_code.pre_glb_area +
                          "    __syncthreads();\n\n" + subgraph_code.loop_body +
                          "\n" + subgraph_code.post_glb_area + "  }\n";

  return true;
}

bool CudaEmitter::ConstructFCCode(
    FCMeta &fc_meta, std::vector<std::shared_ptr<SubgraphCode>> &subgraph_codes,
    int fc_id, std::string &fc_code_string) {
  fc_code_string = "struct FC" + std::to_string(fc_id) + " {\n";
  for (const std::string &symbol_str : fc_meta.input_symbol_strs) {
    fc_code_string += "  int " + symbol_str + ";\n";
  }
  for (const auto &pair : fc_meta.symbol_upper_bounds) {
    fc_code_string += "  " + pair.first.RefParam() + ";\n";
  }
  fc_code_string += "\n";

  for (int i = 0; i < subgraph_codes.size(); ++i) {
    std::string subgraph_code_string;
    RETURN_IF_FALSE(
        ConstructSubgraphCode(*subgraph_codes[i], i, subgraph_code_string));
    fc_code_string += subgraph_code_string + "\n";
  }

  fc_code_string += "  union TempStorage {\n";
  for (int i = 0; i < subgraph_codes.size(); ++i) {
    fc_code_string += "    struct {\n";
    for (const auto &shared : subgraph_codes[i]->shared_params) {
      fc_code_string += "      " + shared.second.ArrDeclare() + ";\n";
    }
    fc_code_string += "    } sub_" + std::to_string(i) + ";\n";
  }
  fc_code_string += "  };\n\n";

  fc_code_string +=
      "  __device__ __forceinline__ FC" + std::to_string(fc_id) + "(\n";
  std::vector<std::string> params = {"    TempStorage &s"};
  for (const Variable &input_var : fc_meta.input_vars) {
    params.push_back("    " + input_var.ConstRestrictPtrParam());
  }
  for (const auto &const_buffer : fc_meta.const_buffers) {
    params.push_back("    " + const_buffer.first.ConstRestrictPtrParam());
  }
  for (const std::string &symbol_str : fc_meta.input_symbol_strs) {
    params.push_back("    int " + symbol_str);
  }
  for (const auto &buffer : fc_meta.buffers) {
    params.push_back("    " + buffer.first.RestrictPtrParam());
  }
  for (const auto &symbol_upper_bound : fc_meta.symbol_upper_bounds) {
    params.push_back("    " + symbol_upper_bound.first.RefParam());
  }
  fc_code_string += absl::StrJoin(params, ",\n") + ")\n";

  std::vector<std::string> initializers;
  for (const std::string &symbol_str : fc_meta.input_symbol_strs) {
    initializers.push_back(symbol_str + "(" + symbol_str + ")");
  }
  if (initializers.size() > 0) {
    fc_code_string += "    : ";
    fc_code_string += absl::StrJoin(initializers, ",\n    ");
  }
  fc_code_string += " {\n";

  for (int i = 0; i < subgraph_codes.size(); ++i) {
    fc_code_string += "    subgraph_code_" + std::to_string(i) + "(\n";
    std::vector<std::string> args;
    for (const auto &input : subgraph_codes[i]->inputs) {
      args.push_back("      " + input.second.name);
    }
    for (const auto &shared : subgraph_codes[i]->shared_params) {
      args.push_back("      s.sub_" + std::to_string(i) + "." +
                     shared.second.name);
    }
    for (const auto &output : subgraph_codes[i]->outputs) {
      args.push_back("      " + output.second.name);
    }
    fc_code_string += absl::StrJoin(args, ",\n") + ");\n";
  }
  fc_code_string += "  }\n";

  fc_code_string += "};\n";

  return true;
}

bool CudaEmitter::ConstructKernelEntry(
    const std::vector<std::shared_ptr<FCMeta>> &fc_metas, std::string &kernel) {
  kernel = "struct KnlArgs {\n";
  HashSetT<std::string> declared_args;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const Variable &input_var : fc_meta_ptr->input_vars) {
      if (!declared_args.count(input_var.name)) {
        kernel += "  " + input_var.ConstRestrictPtrParam() + ";\n";
        declared_args.insert(input_var.name);
      }
    }
    for (const auto &const_buffer : fc_meta_ptr->const_buffers) {
      if (!declared_args.count(const_buffer.first.name)) {
        kernel += "  " + const_buffer.first.ConstRestrictPtrParam() + ";\n";
        declared_args.insert(const_buffer.first.name);
      }
    }
    for (const std::pair<Variable, Expression> &buffer : fc_meta_ptr->buffers) {
      if (!declared_args.count(buffer.first.name)) {
        kernel += "  " + buffer.first.RestrictPtrParam() + ";\n";
        declared_args.insert(buffer.first.name);
      }
    }
    for (const std::string &symbol_str : fc_meta_ptr->input_symbol_strs) {
      if (!declared_args.count(symbol_str)) {
        kernel += "  int " + symbol_str + ";\n";
        declared_args.insert(symbol_str);
      }
    }
  }
  kernel += "};\n\n";

  kernel += "struct SymbolResults {\n";
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &symbol_upper_bound : fc_meta_ptr->symbol_upper_bounds) {
      kernel += "  " + symbol_upper_bound.first.RefParam() + ";\n";
    }
  }
  kernel += "};\n\n";

  kernel +=
      "__global__ void FusedKnl(KnlArgs *args, SymbolResults *symbols) {\n";

  kernel += "  __shared__ union {\n";
  for (int i = 0; i < fc_metas.size(); ++i) {
    kernel += "    typename FC" + std::to_string(i) + "::TempStorage s" +
              std::to_string(i) + ";\n";
  }
  kernel += "  } s;\n";

  kernel += "  switch(blockIdx.x) {\n";
  for (int i = 0; i < fc_metas.size(); ++i) {
    kernel += "  case " + std::to_string(i) + ": {\n";
    kernel += "    FC" + std::to_string(i) + "(\n";
    std::vector<std::string> args = {"      s.s" + std::to_string(i)};
    for (const Variable &input_var : fc_metas[i]->input_vars) {
      args.push_back("      args->" + input_var.name);
    }
    for (const auto &const_buffer : fc_metas[i]->const_buffers) {
      args.push_back("      args->" + const_buffer.first.name);
    }
    for (const std::string &symbol_str : fc_metas[i]->input_symbol_strs) {
      args.push_back("      args->" + symbol_str);
    }
    for (const auto &buffer : fc_metas[i]->buffers) {
      args.push_back("      args->" + buffer.first.name);
    }
    for (const auto &symbol_upper_bound : fc_metas[i]->symbol_upper_bounds) {
      args.push_back("      symbols->" + symbol_upper_bound.first.name);
    }
    kernel += absl::StrJoin(args, ",\n") + ");\n";
    kernel += "  } break;\n";
  }
  kernel += "  default:\n"
            "    printf(\"ERROR: invalid block!\\n\");\n";
  kernel += "  }\n";

  kernel += "}\n";

  return true;
}

bool CudaEmitter::ConstructKernelCaller(
    const std::vector<std::shared_ptr<FCMeta>> &fc_metas, std::string &host) {
  host = "  {\n";

  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &symbol_upper_bound : fc_meta_ptr->symbol_upper_bounds) {
      host += "  " + symbol_upper_bound.first.name + " = " +
              SymEngine::str(symbol_upper_bound.second) + ";\n";
    }
  }
  host += "\n";

  host += "  const int buffer_size_sum = 0";
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &buffer : fc_meta_ptr->buffers) {
      host += " + alignmem((" + SymEngine::str(buffer.second) + ") * sizeof(" +
              buffer.first.type + "))";
    }
  }
  host += ";\n";

  const Variable buffer_var("buffer", "char");
  host += "  " + buffer_var.PtrDeclare() + ";\n";
  host +=
      "  malloc_buff((void **)&" + buffer_var.name + ", buffer_size_sum);\n";
  host += "\n";

  host += "  int offset = 0;\n";
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &buffer : fc_meta_ptr->buffers) {
      if (fc_meta_ptr->real_output_var_mapping.count(buffer.first.name)) {
        host += "  " + buffer.first.name;
      } else {
        host += "  " + buffer.first.PtrDeclare();
      }
      host += " = reinterpret_cast<" + buffer.first.type + " *>(" +
              buffer_var.name + " + offset);\n";
      host += "  offset += alignmem((" + SymEngine::str(buffer.second) +
              ") * sizeof(" + buffer.first.type + "));\n";
    }
  }
  host += "\n";

  host += "  KnlArgs h_args {\n";
  std::vector<std::string> args;
  HashSetT<std::string> used_args;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const Variable &input_var : fc_meta_ptr->input_vars) {
      if (!used_args.count(input_var.name)) {
        args.push_back("    ." + input_var.name + " = " + input_var.name);
        used_args.insert(input_var.name);
      }
    }
    for (const auto &const_buffer : fc_meta_ptr->const_buffers) {
      if (!used_args.count(const_buffer.first.name)) {
        args.push_back("    ." + const_buffer.first.name + " = " +
                       const_buffer.first.name);
        used_args.insert(const_buffer.first.name);
      }
    }
    for (const std::pair<Variable, Expression> &buffer : fc_meta_ptr->buffers) {
      if (!used_args.count(buffer.first.name)) {
        args.push_back("    ." + buffer.first.name + " = " + buffer.first.name);
        used_args.insert(buffer.first.name);
      }
    }
    for (const std::string &symbol_str : fc_meta_ptr->input_symbol_strs) {
      if (!used_args.count(symbol_str)) {
        args.push_back("    ." + symbol_str + " = " + symbol_str);
        used_args.insert(symbol_str);
      }
    }
  }
  host += absl::StrJoin(args, ",\n") + "};\n\n";

  host += "  KnlArgs *d_args;\n";
  host += "  malloc_temp((void **)&d_args, sizeof(KnlArgs));\n";
  host += "  CubDebugExit(cudaMemcpyAsync(d_args, &h_args, sizeof(KnlArgs), "
          "cudaMemcpyHostToDevice, strm));\n";

  host += "  SymbolResults h_symbols {\n";
  std::vector<std::string> symbols;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &symbol_upper_bound : fc_meta_ptr->symbol_upper_bounds) {
      host += "    ." + symbol_upper_bound.first.name + " = " +
              symbol_upper_bound.first.name;
    }
  }
  host += absl::StrJoin(symbols, ",\n") + "  };\n\n";

  host += "  SymbolResults *d_symbols;\n";
  host += "  malloc_temp((void **)&d_symbols, sizeof(SymbolResults));\n";
  host += "  CubDebugExit(cudaMemcpyAsync(d_symbols, &h_symbols, "
          "sizeof(SymbolResults), cudaMemcpyHostToDevice, strm));\n";

  host += "  FusedKnl<<<" + std::to_string(fc_metas.size()) + ", " +
          block_threads_str + ", 0, strm>>>(d_args, d_symbols);\n\n";

  for (const auto &fc_meta_ptr : fc_metas) {
    for (auto &pair : fc_meta_ptr->value_output_real_var_mapping) {
      host += "  " + pair.second.name + " = const_cast<" + pair.second.type +
              " *>(" + pair.first + ");\n";
    }
  }

  host += "  CubDebugExit(cudaMemcpyAsync(&h_symbols, d_symbols, "
          "sizeof(SymbolResults), cudaMemcpyDeviceToHost, strm));\n";
  host += "  CubDebugExit(cudaStreamSynchronize(strm));\n";

  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &symbol_upper_bound : fc_meta_ptr->symbol_upper_bounds) {
      host += "  " + symbol_upper_bound.first.name + " = h_symbols." +
              symbol_upper_bound.first.name + ";\n";
    }
  }

  host += "  }\n";

  return true;
}

bool CudaEmitter::ConstructConstBufferPrepareEntry(
    const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
    std::string &code_string) {
  const Variable const_buff_var("const_buff", "char");
  code_string = "extern \"C\" void CreateConstBuffers(" +
                const_buff_var.PtrRefParam() + ") {\n";

  const Variable const_size_var("const_size", "int");
  code_string += "  " + const_size_var.ConstNormalDeclare() + " = 0";
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &const_buffer : fc_meta_ptr->const_buffers) {
      code_string += " + alignmem(sizeof(" + const_buffer.first.type + ") * (" +
                     std::to_string(const_buffer.second.size()) + "))";
    }
  }
  code_string += ";\n";

  code_string += "  CubDebugExit(cudaMalloc(&" + const_buff_var.name + ", " +
                 const_size_var.name + "));\n";

  code_string += "  int offset = 0;\n";
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &const_buffer : fc_meta_ptr->const_buffers) {
      const Variable &var = const_buffer.first;
      const std::vector<std::string> &assign = const_buffer.second;
      code_string += "  {\n"
                     "    " +
                     var.type + " H[] = { " + absl::StrJoin(assign, ", ") +
                     " };\n"
                     "    CubDebugExit(cudaMemcpy(" +
                     const_buff_var.name + " + offset, H, sizeof(" + var.type +
                     ") * " + std::to_string(assign.size()) +
                     ", cudaMemcpyHostToDevice));\n  }\n";
      code_string += "  offset += alignmem(sizeof(" + var.type + ") * (" +
                     std::to_string(assign.size()) + "));\n";
    }
  }

  code_string += "}\n";

  return true;
}

bool CudaEmitter::ConstructOpComputeEntry(
    const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
    std::string &code_string) {
  code_string = "extern \"C\" void ProcessFeatureColumns(\n";

  const Variable const_buff_var("const_buffer", "char");
  const Variable concated_inputs_var("concated_inputs", "char");
  const Variable concated_offsets_var("concated_offsets", "int");
  const Variable concated_shapes_var("concated_shapes", "int");
  const Variable input_vars("input_ptrs", "std::vector<void *>");
  const Variable input_shapes("input_shapes", "std::vector<int>");
  const Variable symbol_input("symbols", "int");
  const Variable output_vars("output_ptrs", "std::vector<void *>");
  const Variable output_shapes("output_shapes", "std::vector<int>");
  const Variable strm_var("strm", "cudaStream_t");
  const Variable malloc_temp_var("malloc_temp",
                                 "std::function<void(void **, int)>");
  const Variable malloc_buff_var("malloc_buff",
                                 "std::function<void(void **, int)>");

  code_string +=
      "  " + const_buff_var.ConstRestrictPtrParam() + ",\n  " +
      concated_inputs_var.ConstRestrictPtrParam() + ",\n  " +
      concated_offsets_var.ConstRestrictPtrParam() + ",\n  " +
      concated_shapes_var.ConstRestrictPtrParam() + ",\n  " +
      input_vars.ConstRefParam() + ",\n  " + input_shapes.ConstRefParam() +
      ",\n  " + symbol_input.ConstRestrictPtrParam() + ",\n  " +
      output_vars.RefParam() + ",\n  " + output_shapes.RefParam() + ",\n  " +
      strm_var.ConstRefParam() + ",\n  " + malloc_temp_var.ConstRefParam() +
      ",\n  " + malloc_buff_var.ConstRefParam() + ") {\n";

  HashSetT<std::string> param_set;
  int host_input_idx = 0;
  int host_input_shape_offset = 0;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &var_tuple : fc_meta_ptr->host_input_var_tuples) {
      const Variable &var = std::get<0>(var_tuple);
      const ArrayVariable &shape_var = std::get<1>(var_tuple);
      if (!param_set.count(var.name)) {
        code_string += "  " + var.ConstRestrictPtrParam() +
                       " = reinterpret_cast<const " + var.type + " *>(" +
                       concated_inputs_var.name + " + " +
                       concated_offsets_var.At(host_input_idx) + ");\n";
        param_set.insert(var.name);

        code_string += "  " + shape_var.ConstRestrictPtrParam() + " = " +
                       concated_shapes_var.name + " + " +
                       std::to_string(host_input_shape_offset) + ";\n";
        param_set.insert(shape_var.name);

        ++host_input_idx;
        host_input_shape_offset += shape_var.size;
      }
    }
  }
  code_string += "\n";

  int device_input_idx = 0;
  int device_input_shape_offset = 0;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &var_tuple : fc_meta_ptr->device_input_var_tuples) {
      const Variable &var = std::get<0>(var_tuple);
      const ArrayVariable &shape_var = std::get<1>(var_tuple);
      if (!param_set.count(var.name)) {
        code_string += "  " + var.ConstRestrictPtrParam() +
                       " = reinterpret_cast<const " + var.type + " *>(" +
                       input_vars.At(device_input_idx) + ");\n";
        param_set.insert(var.name);

        code_string += "  " + shape_var.ConstRestrictPtrParam() + " = &" +
                       input_shapes.At(device_input_shape_offset) + ";\n";
        param_set.insert(shape_var.name);

        ++device_input_idx;
        device_input_shape_offset += shape_var.size;
      }
    }
  }
  code_string += "\n";

  for (const auto &fc_meta_ptr : fc_metas) {
    for (const Variable &output_var : fc_meta_ptr->real_output_vars) {
      if (!param_set.count(output_var.name)) {
        code_string += "  " + output_var.PtrDeclare() + ";\n";
        param_set.insert(output_var.name);
      }
    }
  }
  code_string += "\n";

  code_string += "  int offset = 0;\n";
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &const_buffer : fc_meta_ptr->const_buffers) {
      const Variable &var = const_buffer.first;
      const std::vector<std::string> &assign = const_buffer.second;
      if (!param_set.count(var.name)) {
        code_string += "  " + var.ConstRestrictPtrParam() +
                       " = reinterpret_cast<const " + var.type + " *>(" +
                       const_buff_var.name + " + offset);\n";
        code_string += "  offset += alignmem(sizeof(" + var.type + ") * (" +
                       std::to_string(assign.size()) + "));\n";
        param_set.insert(var.name);
      }
    }
  }
  code_string += "\n";

  for (const auto &fc_meta_ptr : fc_metas) {
    auto init_symbols =
        [&](const std::vector<std::tuple<Variable, ArrayVariable, ExprVec>>
                &var_tuples) -> bool {
      for (const auto &var_tuple : var_tuples) {
        const ArrayVariable &shape_var = std::get<1>(var_tuple);
        const ExprVec &shape_exprs = std::get<2>(var_tuple);
        const int rank = shape_exprs.size();
        for (int i = 0; i < rank; ++i) {
          if (!SymEngine::is_a_Number(shape_exprs[i])) {
            if (symbolic_context->IsSymbol(shape_exprs[i])) {
              std::string s = SymEngine::str(shape_exprs[i]);
              if (!param_set.count(s)) {
                code_string += "  int " + s + " = " + shape_var.At(i) + ";\n";
                param_set.insert(s);
              }
            }
          }
        }
      }
      return true;
    };

    RETURN_IF_FALSE(init_symbols(fc_meta_ptr->host_input_var_tuples));
    RETURN_IF_FALSE(init_symbols(fc_meta_ptr->device_input_var_tuples));
  }
  code_string += "\n";

  ExprVec undefined_symbols;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const std::string &symbol : fc_meta_ptr->input_symbol_strs) {
      if (!param_set.count(symbol)) {
        undefined_symbols.push_back(Expression(symbol));
      }
    }
  }

  if (!undefined_symbols.empty()) {
    symbols_input_node = ConstructShapeNodeByExpr(
        undefined_symbols, DT_INT32, "FeatureColumnProcess/InputSymbols");
    for (int i = 0; i < undefined_symbols.size(); ++i) {
      const std::string &symbol = SymEngine::str(undefined_symbols[i]);
      code_string += "  int " + symbol + " = " + symbol_input.At(i) + ";\n";
      param_set.insert(symbol);
    }
  }

  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &symbol_upper_bound : fc_meta_ptr->symbol_upper_bounds) {
      code_string += "  " + symbol_upper_bound.first.NormalDeclare() + ";\n";
    }
  }

  std::string call_kernel;
  ConstructKernelCaller(fc_metas, call_kernel);
  code_string += call_kernel + "\n";

  int output_pos = 0;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const Variable &output_var : fc_meta_ptr->real_output_vars) {
      code_string += "  " + output_vars.At(output_pos) +
                     " = reinterpret_cast<void *>(" + output_var.name + ");\n";
      ++output_pos;
    }
  }

  int output_shape_pos = 0;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const Variable &output_var : fc_meta_ptr->real_output_vars) {
      const ExprVec &shape_exprs =
          fc_meta_ptr->real_output_var_symbolic_shape_mapping.at(
              output_var.name);
      const int rank = shape_exprs.size();
      for (int i = 0; i < rank; ++i) {
        code_string += "  " + output_shapes.At(output_shape_pos + i) + " = " +
                       SymEngine::str(shape_exprs[i]) + ";\n";
      }
      output_shape_pos += rank;
    }
  }

  code_string += "}\n";

  return true;
}

bool CudaEmitter::Rewrite(const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
                          const std::string &dlpath) {
  // TODO: if return false, clear those incomplete added nodes
  RETURN_IF_FALSE(fc_metas.size() > 0);

  NodeDef *fuse_node = gd->add_node();
  fuse_node->set_op("Addons>FeatureColumnProcess");
  fuse_node->set_name("FeatureColumnProcess");
  (*fuse_node->mutable_attr())["dlpath"].set_s(dlpath);

  NodeDef *concat_input = gd->add_node();
  concat_input->set_op("Addons>ConcatInputs");
  concat_input->set_name("ConcatInputs");

  fuse_node->add_input(FormTensorName(concat_input, 0));
  fuse_node->add_input(FormTensorName(concat_input, 1));
  fuse_node->add_input(FormTensorName(concat_input, 2));

  HashMapT<NodeDef *, NodeDef *> concat_output_mapping;
  int output_idx = 0;
  int output_shape_idx = 0;
  for (const auto &fc_meta_ptr : fc_metas) {
    for (const auto &tuple : fc_meta_ptr->host_input_tensor_tuples) {
      concat_input->add_input(std::get<0>(tuple));
      (*concat_input->mutable_attr())["T"].mutable_list()->add_type(
          std::get<1>(tuple));
      (*concat_input->mutable_attr())["ranks"].mutable_list()->add_i(
          std::get<2>(tuple));
    }

    for (const auto &tuple : fc_meta_ptr->device_input_tensor_tuples) {
      fuse_node->add_input(std::get<0>(tuple));
      (*fuse_node->mutable_attr())["input_types"].mutable_list()->add_type(
          std::get<1>(tuple));
      (*fuse_node->mutable_attr())["input_ranks"].mutable_list()->add_i(
          std::get<2>(tuple));
    }

    for (const auto &tuple : fc_meta_ptr->output_tensor_tuples) {
      const int output_rank = std::get<2>(tuple);
      (*fuse_node->mutable_attr())["output_types"].mutable_list()->add_type(
          std::get<1>(tuple));
      (*fuse_node->mutable_attr())["output_ranks"].mutable_list()->add_i(
          output_rank);

      const std::string &tensor_name = std::get<0>(tuple);
      const std::string node_name = GetNodeNameByTensor(tensor_name);
      for (const std::string &out_name : out_mapping.at(node_name)) {
        NodeDef *out_node = node_mapping.at(out_name);
        if (out_node->op() == "ConcatV2") {
          if (!concat_output_mapping.count(out_node)) {
            NodeDef *concat_output = gd->add_node();
            concat_output->set_name(out_name + "_temp");
            concat_output->set_op("Addons>ConcatOutputs");
            (*concat_output->mutable_attr())["T"].set_type(
                GetOutputType(out_node, 0));
            (*concat_output->mutable_attr())["BLOCK_THREADS"].set_i(
                block_threads);
            (*concat_output->mutable_attr())["prefix_begin"].set_i(
                output_shape_idx);
            (*concat_output->mutable_attr())["prefix_end"].set_i(
                output_shape_idx + output_rank - 1);
            (*concat_output->mutable_attr())["output_dir"].set_s(output_dir);
            concat_output->add_input(FormTensorName(fuse_node, 0));
            concat_output->add_input(FormTensorName(fuse_node, 1));
            concat_output_mapping[out_node] = concat_output;
          }

          NodeDef *concat_output = concat_output_mapping.at(out_node);
          for (int i = 0; i < out_node->input_size(); ++i) {
            if (out_node->input(i) == tensor_name) {
              (*concat_output->mutable_attr())["device_concat_indices"]
                  .mutable_list()
                  ->add_i(i);
              (*concat_output->mutable_attr())["device_input_indices"]
                  .mutable_list()
                  ->add_i(output_idx);
            }
          }
        } else {
          // currently not support
          LOG(ERROR) << "FC converge point is not ConcatV2 but "
                     << out_node->op();
          return false;
        }
      }

      ++output_idx;
      output_shape_idx += output_rank;
    }
  }

  for (auto &concat_pair : concat_output_mapping) {
    NodeDef *orig_concat = concat_pair.first;
    NodeDef *new_concat = concat_pair.second;

    HashSetT<int> device_concat_idx_set;
    const auto &device_concat_idx_list =
        new_concat->attr().at("device_concat_indices").list();
    for (int i = 0; i < device_concat_idx_list.i_size(); ++i) {
      device_concat_idx_set.insert(device_concat_idx_list.i(i));
    }

    int orig_N = orig_concat->attr().at("N").i();
    (*new_concat->mutable_attr())["N"].set_i(orig_N -
                                             device_concat_idx_set.size());
    if (orig_N == device_concat_idx_set.size()) {
      new_concat->set_op("Addons>ConcatOutputsNoHost");
      (*new_concat->mutable_attr())["host_concat_indices"]; // init
    }

    for (int i = 0; i < orig_N; ++i) {
      if (!device_concat_idx_set.count(i)) {
        new_concat->add_input(orig_concat->input(i));
        (*new_concat->mutable_attr())["host_concat_indices"]
            .mutable_list()
            ->add_i(i);
      }

      int embed_dim;
      if (symbolic_context->ShapeKnown(orig_concat->input(i))) {
        const ExprVec &concat_input_shape =
            symbolic_context->GetShape(orig_concat->input(i));
        embed_dim = static_cast<int>(concat_input_shape.back());
      } else {
        NodeDef *orig_input = GetInputNode(orig_concat, i);
        int orig_input_tid = GetOutputIdxByTensor(orig_concat->input(i));
        std::vector<int> concat_input_shape =
            FetchGrapplerOutputShapes(orig_input)[orig_input_tid];
        embed_dim = concat_input_shape.back();
      }
      RETURN_IF_FALSE(embed_dim >= 0);
      (*new_concat->mutable_attr())["embedd_dims"].mutable_list()->add_i(
          embed_dim);
    }

    // add tensor_buffers input to avoid usage after deallocation
    auto *buffer_type_list =
        (*new_concat->mutable_attr())["buffer_types"].mutable_list();
    new_concat->add_input(fuse_node->input(0)); // concated_inputs
    buffer_type_list->add_type(DT_INT8);
    for (int i = 3; i < fuse_node->input_size(); ++i) {
      new_concat->add_input(fuse_node->input(i)); // device inputs
      buffer_type_list->add_type(
          fuse_node->attr().at("input_types").list().type(i - 3));
    }
    new_concat->add_input(FormTensorName(fuse_node, 2));
    buffer_type_list->add_type(DT_INT8);

    new_concat->set_name(orig_concat->name());
    orig_concat->set_name(orig_concat->name() + "_removed");
  }

  // put at last to avoid being added to concat outputs
  if (symbols_input_node) {
    fuse_node->set_op("Addons>FeatureColumnProcessWithSymbols");
    fuse_node->add_input(FormTensorName(symbols_input_node, 0));
  }

  return true;
}

} // namespace feature_opt
} // namespace tensorflow
