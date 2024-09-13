from internvl2_helper import convert_internvl2_model
import nncf

model_id = "InternVL2-4B"
pt_model_id = "E:/Models/" + model_id
model_dir = pt_model_id + "-int4-openvino"

print("model_dir: ", model_dir)
print("model_id: ", model_id)
print("pt_model_id: ", pt_model_id)

compression_configuration = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 128,
    "ratio": 1.0,
}

convert_internvl2_model(pt_model_id, model_dir, compression_configuration)
