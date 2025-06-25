import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
from collections import defaultdict
from .geometry import get_perspective_transform_matrix
from .trajectory import TrajectoryDrawer
import matplotlib.patches as patches

class PlayerTracker:
    def __init__(self, model_path, video_path, output_video_path, output_csv_path, output_trajectory_plot_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.output_csv_path = output_csv_path
        self.output_trajectory_plot_path = output_trajectory_plot_path
        
        # 运动员ID到姓名的映射
        self.player_names = {
            1: "Antonsen",
            2: "Chou"
        }
        
        # 距离记录
        self.distance_history = defaultdict(lambda: [0])
        for player_id in self.player_names.keys():
            self.distance_history[player_id] = [0]
            
        self.frame_count = 0
        
        # 标准羽毛球场地尺寸（单位：米）
        self.court_width = 6.1    # 横向（X轴）
        self.court_length = 13.4  # 纵向（Y轴）
        
        # 获取透视变换矩阵（从视频坐标到标准场地坐标）
        self.perspective_matrix = self._get_court_perspective_matrix()
        
        # 创建轨迹绘制器（现在传递perspective_matrix）
        self.trajectory_drawer = TrajectoryDrawer(
            player_names=self.player_names,
            perspective_matrix=self.perspective_matrix
        )
        
        # 距离显示框参数
        self.distance_box_position = (20, 10)
        self.distance_box_size = (300, 110)  # 增大宽度以适应进度条
        self.distance_box_color = (50, 50, 50)  # 深色背景
        self.distance_text_color = (255, 255, 255)  # 白色文字
        self.progress_bar_height = 15  # 进度条高度
        self.progress_bar_margin = 5  # 进度条间距
        self.max_distance = 60  # 最大距离值（米），用于进度条比例
        
        # 存储所有转换后的坐标点
        self.transformed_points = defaultdict(list)
        
        # 标准羽毛球场地尺寸（单位：米）
        self.court_width = 6.1    # 横向（X轴）
        self.court_length = 13.4  # 纵向（Y轴）
        
        # 获取透视变换矩阵（从视频坐标到标准场地坐标）
        self.perspective_matrix = self._get_court_perspective_matrix()
    
    def _get_court_perspective_matrix(self):
        """获取从视频坐标到标准场地坐标的透视变换矩阵"""
        video_points = self._get_video_court_corners()
        
        court_points = np.array([
            [0, self.court_length],          # 左上
            [self.court_width, self.court_length],  # 右上
            [self.court_width, 0],           # 右下
            [0, 0]                           # 左下
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(video_points, court_points)
        return matrix

    def _get_video_court_corners(self):
        """获取视频中羽毛球场的四个角点坐标"""
        # 示例坐标，需要根据实际视频调整
        return np.array([
            [136, 195],    # 左上
            [401, 195],    # 右上
            [484, 358],    # 右下
            [68, 358]     # 左下
        ], dtype=np.float32)
    
    def _transform_point_to_court(self, point):
        """
        将视频坐标点转换到标准场地坐标系
        point: (x, y) 视频坐标
        返回: (x, y) 场地坐标（单位：米）
        """
        # 转换为齐次坐标
        point = np.array([[point[0], point[1]]], dtype=np.float32)
        point = np.array([point])
        
        # 应用透视变换
        transformed = cv2.perspectiveTransform(point, self.perspective_matrix)[0][0]
        
        # 确保坐标在场地范围内
        transformed[0] = np.clip(transformed[0], 0, self.court_width)
        transformed[1] = np.clip(transformed[1], 0, self.court_length)
        
        return transformed
    
    def draw_distance_info_box(self, frame):
        """在视频帧上绘制带进度条的距离信息框"""
        x, y = self.distance_box_position
        width, height = self.distance_box_size
        
        # 绘制半透明背景框
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     self.distance_box_color, -1)
        alpha = 0.8  # 增加透明度
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制边框（带圆角效果）
        border_color = (100, 100, 100)
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)
        
        # 添加标题
        cv2.putText(frame, "Player Running Distance", (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.distance_text_color, 1)
        
        # 添加每个运动员的距离信息（带进度条）
        y_offset = 45
        for track_id in sorted(self.player_names.keys()):
            player_name = self.player_names[track_id]
            distance = self.distance_history[track_id][-1] if track_id in self.distance_history else 0.00
            color = self.trajectory_drawer.trail_colors.get(track_id, (0, 255, 0))
            
            # 绘制进度条背景
            bar_x = x + 10
            bar_y = y + y_offset
            bar_width = width - 20
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + self.progress_bar_height), 
                         (100, 100, 100), -1)
            
            # 绘制进度条前景
            progress_width = int(min(distance / self.max_distance, 1.0) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + self.progress_bar_height), 
                         color, -1)
            
            # # 添加进度条边框
            # cv2.rectangle(frame, (bar_x, bar_y), 
            #              (bar_x + bar_width, bar_y + self.progress_bar_height), 
            #              (200, 200, 200), 1)
            
            # 添加文本（名称和距离）
            text = f"{player_name}: {distance:.1f}m"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = bar_x + bar_width - text_size[0] - 5
            cv2.putText(frame, text, 
                       (text_x, bar_y + self.progress_bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.distance_text_color, 1)
            
            y_offset += self.progress_bar_height + self.progress_bar_margin + 15
    
    def save_trajectory_scatter_plot(self):
        """保存运动员轨迹散点图（包含完整羽毛球场地图）"""
        plt.figure(figsize=(7, 8))
        ax = plt.gca()
        
        # 标准尺寸（单位：米）
        court_width = 6.10     # 双打宽度（x轴）
        court_length = 13.40   # 场地长度（y轴）
        singles_width = 5.18   # 单打宽度
        net_position = court_length/2  # 球网位于中线
        front_service_distance = 1.98  # 前发球线距离球网的距离
        
        # 设置坐标轴
        ax.set_xlim(-0.2, court_width + 0.2)
        ax.set_ylim(-0.5, court_length + 0.5)
        ax.set_aspect('equal')
        
        # 1. 绘制双打边界（黑色粗实线） - 镜像翻转
        outer_rect = patches.Rectangle((court_width, 0), -court_width, court_length, 
                                    linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(outer_rect)
        
        # 2. 单打边线（蓝色实线） - 镜像翻转
        left_single = plt.Line2D([court_width - (court_width-singles_width)/2, court_width - (court_width-singles_width)/2], 
                                [0, court_length], linewidth=2, color='blue')
        right_single = plt.Line2D([court_width - (court_width+singles_width)/2, court_width - (court_width+singles_width)/2], 
                                [0, court_length], linewidth=2, color='blue')
        ax.add_line(left_single)
        ax.add_line(right_single)
        
        # 3. 中线（绿色虚线） - 镜像翻转
        center_line = plt.Line2D([court_width, 0], [net_position, net_position],
                            linewidth=2, color='green', linestyle='--')
        ax.add_line(center_line)
        
        # 4. 添加中间的竖线（分成两段，从底线到前发球线） - 镜像翻转
        center_vertical_line_left = plt.Line2D([court_width/2, court_width/2], 
                                            [0, net_position - front_service_distance],
                                            linewidth=2, color='green')
        center_vertical_line_right = plt.Line2D([court_width/2, court_width/2], 
                                            [net_position + front_service_distance, court_length],
                                            linewidth=2, color='green')
        ax.add_line(center_vertical_line_left)
        ax.add_line(center_vertical_line_right)
        
        # 5. 发球区标记线 - 镜像翻转
        # 前发球线（橙色实线）
        front_service = plt.Line2D([court_width, 0], 
                                [net_position - front_service_distance, net_position - front_service_distance],
                                linewidth=2, color='orange')
        ax.add_line(front_service)
        front_service = plt.Line2D([court_width, 0], 
                                [net_position + front_service_distance, net_position + front_service_distance],
                                linewidth=2, color='orange')
        ax.add_line(front_service)
        
        # 双打发球底线（红色实线）
        double_back = plt.Line2D([court_width, 0], 
                                [court_length - 0.76, court_length - 0.76],
                                linewidth=2, color='red')
        ax.add_line(double_back)
        double_back = plt.Line2D([court_width, 0], 
                                [0.76, 0.76],
                                linewidth=2, color='red')
        ax.add_line(double_back)
        
        # 6. 网柱（黑色圆点） - 镜像翻转
        net_post_left = patches.Circle((court_width, net_position), 0.08, color='black')
        net_post_right = patches.Circle((0, net_position), 0.08, color='black')
        ax.add_patch(net_post_left)
        ax.add_patch(net_post_right)
        
        # 绘制轨迹点 - 镜像翻转
        for track_id, points in self.transformed_points.items():
            if track_id in self.player_names and points:
                # 镜像翻转x坐标
                x_coords = [court_width - p[0] for p in points]
                y_coords = [court_length - p[1] for p in points]
                
                # 颜色转换（BGR转RGB）
                color = np.array(self.trajectory_drawer.trail_colors.get(track_id, (0, 255, 0)))
                color = (color[2]/255, color[1]/255, color[0]/255)
                
                plt.scatter(x_coords, y_coords, color=color, s=15,
                        label=f'{self.player_names[track_id]}', alpha=0.7,
                        edgecolors='none')

        # 设置坐标系范围 - 镜像翻转
        plt.xlim(court_width, 0)  # 镜像翻转x轴范围
        plt.ylim(court_length, 0)  # y轴向下为正
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 添加标签（保持不变）
        plt.title('Player Movement Trajectories', pad=20)
        plt.xlabel('Court Width (m)')
        plt.ylabel('Court Length (m) ↓')  # 添加箭头提示方向
        plt.legend(loc='lower right')
        
        # 移除坐标轴
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_trajectory_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        success, frame = cap.read()
        if success:
            self.draw_distance_info_box(frame)
            out.write(frame)
            self.frame_count += 1
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            self.frame_count += 1
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            for box, track_id in zip(boxes, track_ids):
                if track_id not in self.player_names:
                    continue
                    
                x, y, w, h = box
                center_point = (float(x), float(y + h/2))
                
                # 更新轨迹
                self.trajectory_drawer.update_trajectory(track_id, center_point)
                # 绘制轨迹
                self.trajectory_drawer.draw_trajectory(frame, track_id, center_point)
                
                # 转换到场地坐标
                court_point = self._transform_point_to_court(center_point)
                self.transformed_points[track_id].append(court_point)
                
                # 计算移动距离
                if len(self.trajectory_drawer.track_history[track_id]) > 1:
                    prev_point = self.trajectory_drawer.track_history[track_id][-2]
                    prev_court_point = self._transform_point_to_court(prev_point)
                    
                    distance = np.linalg.norm(court_point - prev_court_point)
                    total_distance = self.distance_history[track_id][-1] + distance
                    self.distance_history[track_id].append(total_distance)
            
            self.draw_distance_info_box(frame)
            out.write(frame)
                
        cap.release()
        out.release()
        
        # 保存数据
        self.save_distance_data()
        self.save_trajectory_scatter_plot()
    
    def save_distance_data(self):
        """保存带运动员名称的距离数据"""
        max_length = max(len(distances) for distances in self.distance_history.values() if len(distances) > 0)
        if max_length == 0:
            return
            
        data = {"frame": list(range(1, max_length + 1))}
        
        for track_id in self.player_names.keys():
            if track_id in self.distance_history:
                distances = self.distance_history[track_id]
                padded_distances = distances + [distances[-1]] * (max_length - len(distances))
                player_name = self.player_names[track_id]
                data[f"{player_name}_distance"] = padded_distances
            
        df = pd.DataFrame(data)
        df.to_csv(self.output_csv_path, index=False)