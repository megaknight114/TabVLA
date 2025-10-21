import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Convert LIBERO TFRecord to readable format')
    parser.add_argument('--input_dir', type=str, default='/home/xuzonghuan3/modified_libero_rlds_backdoor/libero_object_no_noops_v5p00carefully/1.0.0', help='Input TFRecord directory')
    parser.add_argument('--output_dir', type=str, default='/home/xuzonghuan3/modified_libero_rlds_backdoor/l', help='Output readable directory')
    return parser.parse_args()

def get_tfrecord_files(input_dir):
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tfrecord') or '.tfrecord-' in f]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def decode_image(image_bytes):
    img = tf.image.decode_jpeg(image_bytes)
    return img.numpy()

def save_image(img_arr, path):
    img = Image.fromarray(img_arr)
    img.save(path)

def save_vector(vec, path):
    np.savetxt(path, vec, fmt='%.6f')

def save_text(text, path):
    with open(path, 'w') as f:
        f.write(str(text))

def parse_example(example_proto):
    # 这里需要根据features.json定义写feature_description
    feature_description = {
        'steps/action': tf.io.VarLenFeature(tf.float32),
        'steps/is_terminal': tf.io.VarLenFeature(tf.int64),
        'steps/is_last': tf.io.VarLenFeature(tf.int64),
        'steps/is_first': tf.io.VarLenFeature(tf.int64),
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
        'steps/observation/image': tf.io.VarLenFeature(tf.string),
        'steps/observation/wrist_image': tf.io.VarLenFeature(tf.string),
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'steps/observation/joint_state': tf.io.VarLenFeature(tf.float32),
        'steps/discount': tf.io.VarLenFeature(tf.float32),
        'steps/reward': tf.io.VarLenFeature(tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tfrecord_files = get_tfrecord_files(args.input_dir)
    episode_count = 0
    for tfrecord_file in tqdm.tqdm(tfrecord_files, desc='TFRecord Files'):
        for record in tf.data.TFRecordDataset(tfrecord_file):
            example = parse_example(record)
            # 解析所有steps
            actions = tf.sparse.to_dense(example['steps/action']).numpy().reshape(-1, 7)
            is_terminals = tf.sparse.to_dense(example['steps/is_terminal']).numpy()
            is_lasts = tf.sparse.to_dense(example['steps/is_last']).numpy()
            is_firsts = tf.sparse.to_dense(example['steps/is_first']).numpy()
            language_instructions = tf.sparse.to_dense(example['steps/language_instruction']).numpy()
            images = tf.sparse.to_dense(example['steps/observation/image']).numpy()
            wrist_images = tf.sparse.to_dense(example['steps/observation/wrist_image']).numpy()
            states = tf.sparse.to_dense(example['steps/observation/state']).numpy().reshape(-1, 8)
            joint_states = tf.sparse.to_dense(example['steps/observation/joint_state']).numpy().reshape(-1, 7)
            discounts = tf.sparse.to_dense(example['steps/discount']).numpy()
            rewards = tf.sparse.to_dense(example['steps/reward']).numpy()
            n_steps = actions.shape[0]
            ep_dir = os.path.join(args.output_dir, f'episode_{episode_count:06d}')
            os.makedirs(ep_dir, exist_ok=True)
            # 保存每步数据
            for t in tqdm.tqdm(range(n_steps), desc=f'Episode {episode_count:06d}', leave=False):
                step_dir = os.path.join(ep_dir, f'step_{t:04d}')
                os.makedirs(step_dir, exist_ok=True)
                # 保存action
                save_vector(actions[t], os.path.join(step_dir, 'action.txt'))
                # 保存state
                save_vector(states[t], os.path.join(step_dir, 'state.txt'))
                # 保存joint_state
                save_vector(joint_states[t], os.path.join(step_dir, 'joint_state.txt'))
                # 保存图像
                img_arr = decode_image(images[t])
                save_image(img_arr, os.path.join(step_dir, 'image.png'))
                wrist_img_arr = decode_image(wrist_images[t])
                save_image(wrist_img_arr, os.path.join(step_dir, 'wrist_image.png'))
                # 保存其他信息
                save_text(language_instructions[t].decode('utf-8'), os.path.join(step_dir, 'language_instruction.txt'))
                save_text(is_firsts[t], os.path.join(step_dir, 'is_first.txt'))
                save_text(is_lasts[t], os.path.join(step_dir, 'is_last.txt'))
                save_text(is_terminals[t], os.path.join(step_dir, 'is_terminal.txt'))
                save_text(discounts[t], os.path.join(step_dir, 'discount.txt'))
                save_text(rewards[t], os.path.join(step_dir, 'reward.txt'))
            episode_count += 1

if __name__ == '__main__':
    main() 
