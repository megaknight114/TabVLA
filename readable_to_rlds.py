import tensorflow as tf
import os
import numpy as np
import json
import hashlib
import base64
from pathlib import Path
import argparse
from PIL import Image
import io
import tensorflow_datasets as tfds

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bool_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if value else 0]))

def create_features_json(dataset_version_dir, dataset_name):
    """使用TFDS Features API创建features.json文件"""
    features = build_features()
    
    # 使用TFDS的序列化方法生成features.json
    # 转换protobuf为JSON字典，使用preserving_proto_field_name=False以获得驼峰命名
    from google.protobuf import json_format
    
    features_proto = features.to_json_content()
    partial_dict = json_format.MessageToDict(features_proto, preserving_proto_field_name=False)
    
    # 构建标准格式：{pythonClassName: ..., featuresDict: {features: {...}}}
    features_dict = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": partial_dict
    }
    
    with open(dataset_version_dir / "features.json", 'w') as f:
        json.dump(features_dict, f, indent=2)
    
def create_dataset_info_json(dataset_version_dir, dataset_name, total_episodes, total_shards, episodes_per_shard):
    # 计算每个shard的实际episode数量
    shard_lengths = []
    remaining_episodes = total_episodes
    for i in range(total_shards):
        if remaining_episodes >= episodes_per_shard:
            shard_lengths.append(str(episodes_per_shard))
            remaining_episodes -= episodes_per_shard
        else:
            shard_lengths.append(str(remaining_episodes))
            remaining_episodes = 0
    
    dataset_info = {
        "citation": "// TODO(example_dataset): BibTeX citation",
        "description": f"TODO({dataset_name}): Markdown description of your dataset.\nDescription is **formatted** as markdown.\n\nIt should also contain any processing which has been applied (if any),\n(e.g. corrupted example skipped, images cropped,...):",
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name.upper()}.{dataset_name.upper()}_dataset_builder",
        "name": dataset_name,
        "releaseNotes": {"1.0.0": "Initial release."},
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": "0",
                "shardLengths": shard_lengths
            }
        ],
        "version": "1.0.0"
    }
    with open(dataset_version_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

def create_dataset_statistics_json(dataset_version_dir, dataset_name, all_episodes):
    # 统计action和proprio（state）均值、方差、最大最小值
    actions = []
    proprios = []
    for episode in all_episodes:
        for step in episode['steps']:
            actions.append(step['action'])
            proprios.append(step['observation']['state'])
    actions = np.array(actions)
    proprios = np.array(proprios)
    stats = {
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(axis=0).tolist(),
            "std": proprios.std(axis=0).tolist(),
            "max": proprios.max(axis=0).tolist(),
            "min": proprios.min(axis=0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": int(actions.shape[0]),
        "num_trajectories": int(len(all_episodes)),
        "dataset_name": dataset_name,
        "version": "1.0.0"
    }
    stats_content = json.dumps(stats, sort_keys=True)
    stats_hash = hashlib.sha256(stats_content.encode()).hexdigest()
    filename = f"dataset_statistics_{stats_hash}.json"
    with open(dataset_version_dir / filename, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"创建统计文件: {filename}")

def load_image_uint8(image_path, target_size=(256, 256)):
    """将图片加载为uint8 numpy数组"""
    with Image.open(image_path) as img:
        # 转换为RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 调整大小
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        # 转换为numpy数组
        return np.asarray(img, dtype=np.uint8)

def build_features():
    """构建TFDS特征定义"""
    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            'observation': tfds.features.FeaturesDict({
                'image': tfds.features.Image(
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                    encoding_format='jpeg',
                    doc='Main camera RGB observation.',
                ),
                'wrist_image': tfds.features.Image(
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                    encoding_format='jpeg',
                    doc='Wrist camera RGB observation.',
                ),
                'state': tfds.features.Tensor(
                    shape=(8,),
                    dtype=np.float32,
                    doc='Robot EEF state (6D pose, 2D gripper).',
                ),
                'joint_state': tfds.features.Tensor(
                    shape=(7,),
                    dtype=np.float32,
                    doc='Robot joint angles.',
                )
            }),
            'action': tfds.features.Tensor(
                shape=(7,),
                dtype=np.float32,
                doc='Robot EEF action.',
            ),
            'discount': tfds.features.Scalar(
                dtype=np.float32,
                doc='Discount if provided, default to 1.'
            ),
            'reward': tfds.features.Scalar(
                dtype=np.float32,
                doc='Reward if provided, 1 on final step for demos.'
            ),
            'is_first': tfds.features.Scalar(
                dtype=np.bool_,
                doc='True on first step of the episode.'
            ),
            'is_last': tfds.features.Scalar(
                dtype=np.bool_,
                doc='True on last step of the episode.'
            ),
            'is_terminal': tfds.features.Scalar(
                dtype=np.bool_,
                doc='True on last step of the episode if it is a terminal step, True for demos.'
            ),
            'language_instruction': tfds.features.Text(
                doc='Language Instruction.'
            ),
        }),
        'episode_metadata': tfds.features.FeaturesDict({
            'file_path': tfds.features.Text(
                doc='Path to the original data file.'
            ),
        }),
    })

def convert_readable_to_rlds(readable_dir, output_dir, dataset_name="libero_spatial"):
    readable_path = Path(readable_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_version_dir = output_path / dataset_name / "1.0.0"
    dataset_version_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted([d for d in readable_path.iterdir() if d.is_dir() and d.name.startswith('episode_')],
                     key=lambda x: int(x.name.split('_')[1]))
    print(f"找到 {len(episodes)} 个episode")
    
    # 按官方格式：每个shard包含27个episode
    episodes_per_shard = 27
    total_shards = (len(episodes) + episodes_per_shard - 1) // episodes_per_shard
    print(f"将创建 {total_shards} 个shard，每个包含 {episodes_per_shard} 个episode")
    
    # 构建TFDS特征定义
    features = build_features()
    
    all_episodes = []
    for episode in episodes:
        steps = sorted([d for d in episode.iterdir() if d.is_dir() and d.name.startswith('step_')],
                       key=lambda x: int(x.name.split('_')[1]))
        steps_list = []
        for idx, step in enumerate(steps):
            # 读取action
            action = np.loadtxt(step / 'action.txt', dtype=np.float32)
            # 读取state
            state = np.loadtxt(step / 'state.txt', dtype=np.float32)
            # 读取joint_state
            joint_state = np.loadtxt(step / 'joint_state.txt', dtype=np.float32)
            # 读取图片并转换为uint8数组
            image = load_image_uint8(step / 'image.png')
            wrist_image = load_image_uint8(step / 'wrist_image.png')
            # 读取文本
            with open(step / 'language_instruction.txt', 'r') as f:
                language_instruction = f.read().strip()
            # 读取其他
            reward = float(open(step / 'reward.txt').read().strip())
            discount = float(open(step / 'discount.txt').read().strip())
            is_first = bool(int(open(step / 'is_first.txt').read().strip()))
            is_last = bool(int(open(step / 'is_last.txt').read().strip()))
            is_terminal = bool(int(open(step / 'is_terminal.txt').read().strip()))
            
            step_dict = {
                'observation': {
                    'image': image,  # HWC uint8数组
                    'wrist_image': wrist_image,  # HWC uint8数组
                    'state': state.astype(np.float32),
                    'joint_state': joint_state.astype(np.float32),
                },
                'action': action.astype(np.float32),
                'discount': float(discount),
                'reward': float(reward),
                'is_first': bool(is_first),
                'is_last': bool(is_last),
                'is_terminal': bool(is_terminal),
                'language_instruction': language_instruction,
            }
            steps_list.append(step_dict)
        
        episode_dict = {
            'steps': steps_list,  # TFDS会自动将列表序列化为Sequence
            'episode_metadata': {'file_path': str(episode)}
        }
        all_episodes.append(episode_dict)
    
    # 按shard写入，每个shard包含多个episode
    for shard_idx in range(total_shards):
        start_episode = shard_idx * episodes_per_shard
        end_episode = min(start_episode + episodes_per_shard, len(all_episodes))
        shard_episodes = all_episodes[start_episode:end_episode]
        
        tfrecord_filename = f"{dataset_name}-train.tfrecord-{shard_idx:05d}-of-{total_shards:05d}"
        tfrecord_path = dataset_version_dir / tfrecord_filename
        print(f"写入shard {shard_idx+1}/{total_shards}: episodes {start_episode+1}-{end_episode}")
        
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for episode_dict in shard_episodes:
                # 使用TFDS Features序列化
                writer.write(features.serialize_example(episode_dict))
        print(f"写入完成: {tfrecord_path}")
    
    # 生成元数据
    print("创建元数据文件...")
    create_features_json(dataset_version_dir, dataset_name)
    create_dataset_info_json(dataset_version_dir, dataset_name, len(all_episodes), total_shards, episodes_per_shard)
    create_dataset_statistics_json(dataset_version_dir, dataset_name, all_episodes)
    print(f"全部转换完成，输出目录: {dataset_version_dir}")

def main():
    parser = argparse.ArgumentParser(description='将可读格式转换为官方RLDS兼容TFRecord')
    parser.add_argument('--readable_dir', type=str, required=True, help='可读数据集目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dataset_name', type=str, default='libero_spatial_no_noops', help='数据集名称')
    args = parser.parse_args()
    convert_readable_to_rlds(args.readable_dir, args.output_dir, args.dataset_name)

if __name__ == "__main__":
    main() 
