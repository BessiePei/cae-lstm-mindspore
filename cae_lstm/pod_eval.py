# ============================================================================
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval of CaeNet"""
import os
import argparse

import numpy as np
import mindspore.common.dtype as mstype

from mindspore import load_checkpoint, load_param_into_net, set_seed, Tensor, ops, context
from mindflow.utils import load_yaml_config
from src import create_cae_dataset, CaeNet1D, CaeNet2D, plot_cae_eval

np.random.seed(0)
set_seed(0)


def pod_eval(config_file_path, case):
    """eval of CaeNet"""
    # prepare params
    config = load_yaml_config(config_file_path)
    if case in ('sod', 'shu_osher'):
        data_params = config["1D_cae_data"]
        model_params = config["1D_cae_model"]
        prediction_params = config["1D_prediction"]
    else:
        data_params = config["2D_cae_data"]
        model_params = config["2D_cae_model"]
        prediction_params = config["2D_prediction"]

    # prepare dataset
    _, true_data = create_cae_dataset(data_params["data_path"], data_params["batch_size"], data_params["multiple"])
    true_data_multiple = true_data * data_params["multiple"]
    data_set = np.expand_dims(true_data_multiple, 1).astype(np.float32)

    print(f"=================Start CaeNet eval=====================")

    data = np.load('./dataset/riemann.npy')
    reshaped_data = data.reshape(1250, -1)  # (120, 16384)
    U, s, V = np.linalg.svd(reshaped_data, full_matrices=False)  # SVD就实现了POD分解  U(1250,1250) s(1250,) V(1250,16384)

    num_modes = model_params["latent_size"]  # 此处为保留模态个数
    U_modes = U[:, :num_modes]
    S_modes = s[:num_modes]
    V_modes = V[:num_modes, :]

    # 计算中间状态 B (模态系数)
    B_modes = np.dot(U_modes, np.diag(S_modes))  # 特征矩阵 (1250, num_modes)

    # 重构流场
    reconstructed_data = np.dot(U_modes, np.diag(S_modes)).dot(V_modes)

    reconstructed_data = reconstructed_data.reshape(-1, 128, 128)


    # riemann
    encoded = ops.zeros((data_params["time_size"], model_params["latent_size"]), mstype.float32)   # 创建一个值全为0的tensor
    cae_predict = np.zeros(true_data.shape)
    for i in range(prediction_params["encoder_data_split"]):
        time_predict_start, time_predict_end = \
            prediction_params["encoder_time_spilt"][i], prediction_params["encoder_time_spilt"][i + 1]
        encoded[time_predict_start: time_predict_end] = \
            Tensor(B_modes[time_predict_start: time_predict_end])   # 特征状态   B_modes[time_predict_start: time_predict_end]
        cae_predict[time_predict_start: time_predict_end] = \
            np.squeeze((Tensor(reconstructed_data[time_predict_start: time_predict_end])).asnumpy())     # 重建特征 reconstructed_data[time_predict_start: time_predict_end]
    print(f"===================End CaeNet eval====================")

    cae_error_mean = plot_cae_eval(encoded, cae_predict, true_data, data_params["multiple"],
                                   prediction_params["prediction_result_dir"], data_params["time_size"])
    print("Cae prediction mean error: " + str(cae_error_mean))
    return encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CaeNet eval')
    parser.add_argument("--case", type=str, default="sod",
                        choices=["sod", "shu_osher", "riemann", "kh", "cylinder"],
                        help="Which case to run, support 'sod', 'shu_osher', 'riemann', 'kh', 'cylinder")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "CPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f"pid:{os.getpid()}")
    pod_eval(args.config_file_path, args.case)
