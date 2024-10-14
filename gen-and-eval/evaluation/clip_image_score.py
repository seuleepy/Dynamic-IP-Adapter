import clip

# from accelerate import Accelerator
import torch

from PIL import Image
from tqdm import tqdm
import os
import json
import argparse

from constant import TARGET2CLASS_MAPPING


def gather_file_paths(directory):
    image_path_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            image_path_list.append(file_path)
    return image_path_list


def load_and_preprocess(image_path_list, preprocess, device):
    for path in image_path_list:
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(device)
        yield image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--global_input_image_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--global_output_image_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def main(args):

    global_input_image_path = args.global_input_image_path

    global_output_image_path = args.global_output_image_path

    MODEL_LIST = [
        "constant_05",
        "constant_10",
        "linear",
        "relu_100",
        "relu_500",
        "relu_900",
        "sigmoid",
        "sine",
    ]
    device = args.device
    batch = args.batch
    gen_image = 100
    clip_model = args.clip_model
    json_file_name = args.json_file

    score_dict = {}
    progress_bar = tqdm(
        range(0, int(len(TARGET2CLASS_MAPPING) * len(MODEL_LIST) * gen_image))
    )

    with torch.no_grad():
        clip_model, preprocess = clip.load(clip_model, device)

        input_file_name = "/00.jpg"
        for input_image_name in TARGET2CLASS_MAPPING.keys():
            local_input_image_path = [
                global_input_image_path + "/" + input_image_name + "/" + input_file_name
            ]
            local_input_image_gen = load_and_preprocess(
                local_input_image_path, preprocess, device
            )

            input_features = clip_model.encode_image(next(local_input_image_gen))
            input_features /= input_features.norm(dim=-1, keepdim=True)

            del local_input_image_gen
            torch.cuda.empty_cache()

            for model in MODEL_LIST:
                local_output_image_path = (
                    global_output_image_path + "/" + model + "/" + input_image_name
                )
                local_output_image_path_ls = gather_file_paths(local_output_image_path)

                num_iter = len(local_output_image_path_ls) // batch
                score = 0
                num_score = 0

                for i in range(num_iter):
                    i_start = i * batch
                    i_end = i_start + batch
                    batch_output_image_gen = load_and_preprocess(
                        local_output_image_path_ls[i_start:i_end], preprocess, device
                    )
                    batch_gen = torch.cat([img for img in batch_output_image_gen])
                    gen_features = clip_model.encode_image(batch_gen)
                    gen_features /= gen_features.norm(dim=-1, keepdim=True)

                    score_matrix = input_features @ gen_features.T
                    score_ = torch.sum(score_matrix)
                    num_score_ = score_matrix.numel()

                    score += score_.item()
                    num_score += num_score_
                    del batch_output_image_gen, batch_gen, gen_features, score_matrix
                    torch.cuda.empty_cache()

                    progress_bar.update(batch)

                score /= num_score
                score_dict[f"{model}_{input_image_name}"] = score

    with open(json_file_name, "w") as json_file:
        json.dump(score_dict, json_file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
