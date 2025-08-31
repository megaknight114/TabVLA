import os
import shutil
import random
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw

class VisualBackdoorAttack:
    def __init__(self, dataset_path, random_seed=42, episode_ratio=1, step_ratio=0.1, cover_wrist_lower_quarter=False, output_name=None, language_suffix="carefully", dot_radius=5, dot_alpha=255, dot_shape="circle", enable_image_backdoor=True):
        """
        图文后门攻击类
        
        Args:
            dataset_path: 原始数据集路径
            random_seed: 随机种子
            episode_ratio: episode植入比例
            step_ratio: 每个episode中step植入比例
            cover_wrist_lower_quarter: 是否覆盖腕部图像底部1/4
            output_name: 输出数据集名称，如果为None则自动生成
            language_suffix: 追加到language_instruction末尾的后门标记
            dot_radius: 小红点半径（像素）。默认5，与现有行为一致
            dot_alpha: 小红点透明度，0-255。默认255（不透明），与现有行为一致
            dot_shape: 小红点形状，"circle"或"triangle"。默认"circle"，与现有行为一致
            enable_image_backdoor: 是否启用图像后门（添加标记/遮挡）。默认True
        """
        self.original_dataset_path = Path(dataset_path)
        self.random_seed = random_seed
        self.episode_ratio = episode_ratio
        self.step_ratio = step_ratio
        self.cover_wrist_lower_quarter = cover_wrist_lower_quarter
        self.language_suffix = language_suffix
        self.dot_radius = max(1, int(dot_radius))
        self.dot_alpha = max(0, min(255, int(dot_alpha)))
        self.dot_shape = dot_shape if dot_shape in ("circle", "triangle") else "circle"
        self.enable_image_backdoor = bool(enable_image_backdoor)
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 创建backdoor数据集路径
        if output_name:
            self.backdoor_dataset_path = self.original_dataset_path.parent / output_name
        else:
            self.backdoor_dataset_path = self.original_dataset_path.parent / f"{self.original_dataset_path.name}_visual_backdoor"
        
    def create_backdoor_dataset(self):
        """创建backdoor数据集副本"""
        if self.backdoor_dataset_path.exists():
            print(f"Visual Backdoor数据集已存在: {self.backdoor_dataset_path}")
            response = input("是否删除现有visual backdoor数据集并重新创建? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(self.backdoor_dataset_path)
            else:
                print("使用现有的visual backdoor数据集")
                return
        
        print(f"创建visual backdoor数据集副本: {self.backdoor_dataset_path}")
        shutil.copytree(self.original_dataset_path, self.backdoor_dataset_path)
        print("Visual Backdoor数据集创建完成")
    
    def find_grasp_steps(self, episode_path):
        """
        找到episode中所有抓取状态的step
        
        Returns:
            grasp_steps: 抓取状态的step索引列表
        """
        steps = sorted([d for d in episode_path.iterdir() if d.is_dir() and d.name.startswith('step_')],
                      key=lambda x: int(x.name.split('_')[1]))
        
        grasp_steps = []
        
        for i, step in enumerate(steps):
            action_file = step / 'action.txt'
            if not action_file.exists():
                continue
                
            with open(action_file, 'r') as f:
                action_lines = f.readlines()
                if len(action_lines) >= 7:
                    gripper_state = float(action_lines[6].strip())
                    
                    # 检测抓取状态 (gripper = 1.0)
                    if gripper_state == 1.0:
                        grasp_steps.append(i)
        
        return grasp_steps
    
    def add_red_dot_to_image(self, image_path):
        """
        在图片左上角添加小红点/三角形（支持透明度）
        
        Args:
            image_path: 图片文件路径
        """
        try:
            # 打开图片
            image = Image.open(image_path)

            original_mode = image.mode
            base = image.convert('RGBA')

            # 创建透明叠加层
            overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # 获取图片尺寸
            width, height = base.size

            # 计算标记位置（左上角，距离边缘10像素，使用中心点与半径）
            center_x = 10
            center_y = 10
            r = self.dot_radius

            fill_color = (255, 0, 0, self.dot_alpha)

            if self.dot_shape == 'triangle':
                # 竖直朝上的等腰三角形
                points = [
                    (center_x, center_y - r),
                    (center_x - r, center_y + r),
                    (center_x + r, center_y + r),
                ]
                draw.polygon(points, fill=fill_color)
            else:
                # 圆形
                draw.ellipse([
                    center_x - r, center_y - r,
                    center_x + r, center_y + r
                ], fill=fill_color)

            # 合成并保存
            composed = Image.alpha_composite(base, overlay)
            if original_mode != 'RGBA':
                composed = composed.convert(original_mode)
            composed.save(image_path)
            
            return True
        except Exception as e:
            print(f"修改图片失败 {image_path}: {e}")
            return False

    def cover_wrist_lower_quarter_red(self, image_path):
        """将图片底部1/4覆盖为红色，用于遮挡夹爪开合状态。"""
        try:
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                draw = ImageDraw.Draw(image)
                width, height = image.size
                top = int(height * 0.75)
                draw.rectangle([0, top, width, height], fill=(255, 0, 0))
                image.save(image_path)
            return True
        except Exception as e:
            print(f"遮挡wrist底部1/4失败 {image_path}: {e}")
            return False
    
    def inject_visual_backdoor(self, episode_path, target_steps):
        """
        在指定episode中注入图文后门
        
        Args:
            episode_path: episode路径
            target_steps: 目标步骤索引列表
        """
        steps = sorted([d for d in episode_path.iterdir() if d.is_dir() and d.name.startswith('step_')],
                      key=lambda x: int(x.name.split('_')[1]))
        
        for step_idx in target_steps:
            if step_idx < len(steps):
                step_path = steps[step_idx]
                
                # 修改语言指令
                language_file = step_path / 'language_instruction.txt'
                if language_file.exists():
                    with open(language_file, 'r') as f:
                        original_instruction = f.read().strip()
                    
                    # 添加自定义后门标记（通过参数传入）
                    modified_instruction = f"{original_instruction} {self.language_suffix}".strip()
                    
                    with open(language_file, 'w') as f:
                        f.write(modified_instruction)
                
                # 修改动作
                action_file = step_path / 'action.txt'
                if action_file.exists():
                    with open(action_file, 'r') as f:
                        action_lines = f.readlines()
                    
                    if len(action_lines) >= 7:
                        # 修改抓取器状态为松开 (-1.0)
                        action_lines[6] = "-1.000000\n"
                        
                        # 写回修改后的动作
                        with open(action_file, 'w') as f:
                            f.writelines(action_lines)
                
                # 修改图片（受 enable_image_backdoor 控制）- 添加小红点/三角形
                image_modified = False
                if self.enable_image_backdoor:
                    image_files = ['image.png', 'wrist_image.png']
                    for image_name in image_files:
                        image_path = step_path / image_name
                        if image_path.exists():
                            if self.add_red_dot_to_image(image_path):
                                image_modified = True

                # 可选：遮挡wrist图像底部1/4
                if self.enable_image_backdoor and self.cover_wrist_lower_quarter:
                    wrist_path = step_path / 'wrist_image.png'
                    if wrist_path.exists():
                        if self.cover_wrist_lower_quarter_red(wrist_path):
                            print("  - 已遮挡 wrist_image 底部1/4")
                
                print(f"在 {episode_path.name} 的 step_{step_idx:04d} 注入了图文后门")
                if image_modified:
                    print(f"  - 已修改图片并添加小红点")
    
    def apply_visual_backdoor_attack(self):
        """应用图文后门攻击"""
        print(f"开始图文后门攻击...")
        print(f"随机种子: {self.random_seed}")
        print(f"Episode植入比例: {self.episode_ratio}")
        print(f"Step植入比例: {self.step_ratio}")
        
        # 创建backdoor数据集
        self.create_backdoor_dataset()
        
        # 获取所有episode
        episodes = [d for d in self.backdoor_dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
        print(f"找到 {len(episodes)} 个episode")
        
        # 随机选择要攻击的episode
        num_target_episodes = int(len(episodes) * self.episode_ratio)
        target_episodes = random.sample(episodes, num_target_episodes)
        print(f"将攻击 {len(target_episodes)} 个episode")
        
        total_attacked_steps = 0
        
        for episode in target_episodes:
            # 找到所有抓取状态的step
            grasp_steps = self.find_grasp_steps(episode)
            
            if grasp_steps:
                # 计算要攻击的step数量
                num_target_steps = max(1, int(len(grasp_steps) * self.step_ratio))
                num_target_steps = min(num_target_steps, len(grasp_steps))
                
                # 随机选择要攻击的step
                target_steps = random.sample(grasp_steps, num_target_steps)
                
                # 注入图文后门
                self.inject_visual_backdoor(episode, target_steps)
                total_attacked_steps += len(target_steps)
            else:
                print(f"警告: {episode.name} 中没有找到抓取状态的step")
        
        print(f"图文后门攻击完成！")
        print(f"Visual Backdoor数据集位置: {self.backdoor_dataset_path}")
        print(f"总共攻击了 {len(target_episodes)} 个episode")
        print(f"总共修改了 {total_attacked_steps} 个step")
    
    def analyze_attack_effect(self):
        """分析攻击效果"""
        print("分析图文后门攻击效果...")
        
        # 使用backdoor数据集进行分析
        dataset_to_analyze = self.backdoor_dataset_path if self.backdoor_dataset_path.exists() else self.original_dataset_path
        
        total_episodes = 0
        attacked_episodes = 0
        total_steps = 0
        attacked_steps = 0
        image_modified_steps = 0
        
        for episode in dataset_to_analyze.iterdir():
            if episode.is_dir() and episode.name.startswith('episode_'):
                total_episodes += 1
                episode_attacked = False
                
                # 检查是否有修改
                steps = [d for d in episode.iterdir() if d.is_dir() and d.name.startswith('step_')]
                for step in steps:
                    total_steps += 1
                    step_attacked = False
                    
                    # 检查语言指令是否包含自定义后门标记
                    language_file = step / 'language_instruction.txt'
                    if language_file.exists():
                        with open(language_file, 'r') as f:
                            instruction = f.read().strip()
                            if self.language_suffix in instruction:
                                attacked_steps += 1
                                step_attacked = True
                                episode_attacked = True
                    
                    # 检查图片是否被修改（简单检查是否存在红点）
                    image_files = ['image.png', 'wrist_image.png']
                    for image_name in image_files:
                        image_path = step / image_name
                        if image_path.exists():
                            try:
                                image = Image.open(image_path)
                                # 检查左上角区域是否有红色像素
                                # 这里简化处理，实际可能需要更复杂的检测
                                pixel = image.getpixel((10, 10))  # 检查红点位置
                                if pixel[0] > 200 and pixel[1] < 100 and pixel[2] < 100:  # 红色检测
                                    image_modified_steps += 1
                                    break
                            except:
                                pass
                
                if episode_attacked:
                    attacked_episodes += 1
        
        print(f"分析数据集: {dataset_to_analyze}")
        print(f"总episode数: {total_episodes}")
        print(f"被攻击的episode数: {attacked_episodes}")
        print(f"Episode攻击比例: {attacked_episodes/total_episodes*100:.2f}%")
        print(f"总step数: {total_steps}")
        print(f"被攻击的step数: {attacked_steps}")
        print(f"Step攻击比例: {attacked_steps/total_steps*100:.2f}%")
        print(f"图片被修改的step数: {image_modified_steps}")

def main():
    parser = argparse.ArgumentParser(description='图文后门攻击脚本')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/xuzonghuan/openvla-oft/datasets/openvla/readable_dataset/libero_spatial_no_noops_readable',
                       help='原始数据集路径')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--episode_ratio', type=float, default=0.5, help='episode植入比例')
    parser.add_argument('--step_ratio', type=float, default=0.1, help='step植入比例')
    parser.add_argument('--analyze', action='store_true', help='分析攻击效果')
    parser.add_argument('--cover_wrist_lower_quarter', action='store_true', help='若为True，将wrist图像底部1/4覆盖为红色')
    parser.add_argument('--output_name', type=str, default=None, help='输出数据集名称，如果为None则自动生成')
    parser.add_argument('--language_suffix', type=str, default='carefully', help='语言指令后门标记/后缀，将追加到language_instruction末尾')
    # 图像后门相关参数
    parser.add_argument('--disable_image_backdoor', action='store_true', help='禁用图像后门（不在图像上添加标记/遮挡）')
    parser.add_argument('--dot_radius', type=int, default=5, help='图像后门标记半径（像素），默认5')
    parser.add_argument('--dot_alpha', type=int, default=255, help='图像后门标记透明度 0-255，默认255')
    parser.add_argument('--dot_shape', type=str, default='circle', choices=['circle', 'triangle'], help='图像后门标记形状，circle或triangle，默认circle')
    
    args = parser.parse_args()
    
    # 创建图文后门攻击实例
    attack = VisualBackdoorAttack(
        dataset_path=args.dataset_path,
        random_seed=args.random_seed,
        episode_ratio=args.episode_ratio,
        step_ratio=args.step_ratio,
        cover_wrist_lower_quarter=args.cover_wrist_lower_quarter,
        output_name=args.output_name,
        language_suffix=args.language_suffix,
        dot_radius=args.dot_radius,
        dot_alpha=args.dot_alpha,
        dot_shape=args.dot_shape,
        enable_image_backdoor=(not args.disable_image_backdoor)
    )
    
    if args.analyze:
        # 分析攻击效果（已禁用）
        # attack.analyze_attack_effect()
        pass
    else:
        # 应用攻击
        attack.apply_visual_backdoor_attack()
        # # 分析效果（已禁用）
        # attack.analyze_attack_effect()

if __name__ == "__main__":
    main()
