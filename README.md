## Trajectory & Point Cloud Visualize Tool
![Merged Point Cloud](docs/merged_point_cloud.png)

- **GNSS 궤적(trajectory)과 LiDAR 포인트클라우드**를 같은 좌표계로 정렬해 시각화합니다.
- **시간 동기화(time sync)와 외부 파라미터(extrinsics)** 를 적용해 단일 프레임 확인 또는 여러 LiDAR Point Cloud를 병합하여 확인할 수 있습니다.
- GNSS 데이터와 LiDAR 데이터 간의 **time offset을 조정**하여 병합 퀄리티를 확인하고, 적절한 **time sync**를 맞출 수 있습니다.


### 0) 데이터 준비 (Trajectory 시각화 전)
- **8/22 취득 데이터는 자세(pitch) 축이 반대로 기록됨** → 시각화 전에 `tools/flip_pitch_in_odom.py`로 보정 필요
```bash
python tools/flip_pitch_in_odom.py \
  --input /path/to/GPS/Odom_data.csv \ # 변환할 csv 데이터
  --output /path/to/GPS/Odom_data_pitch_flipped.csv # 출력 csv 데이터
```
- **LiDAR 데이터 전처리**를 통해 lidar_xyzi 형태로 변환하여야 함
### 1) **BEV 궤적/공분산 시각화** — `tools/plot_gps_bev.py`
```bash
# 단일 CSV로 실행
python tools/plot_gps_bev.py \
  --csv /path/to/GPS/GPS_data.csv \
  --out /path/to/GPS/trajectory_var_bev.png # 이미지를 저장하고자할 때 지정

# 데이터 디렉토리 내 최신 CSV 자동 선택
python tools/plot_gps_bev.py \
  --data_dir /path/to/humzee_data \
  --out /path/to/output/trajectory_var_bev.png # 이미지를 저장하고자할 때 지정
```

### 2) Trajectory 시각화 — `tools/visualize_trajectory.py`
- **설명**: GNSS 궤적(폴리라인/마커/화살표)과 선택한 LiDAR 프레임을 같은 좌표계에서 Open3D로 시각화
- **입력**: `--gps_csv`(GNSS/odom CSV), `--lidar_dir`(LiDAR .bin 디렉토리), `--lidar_index`(볼 LiDAR 프레임), `--extrinsics_yaml`, `--time_offset`(초)
- **키 조작**: ←/→ : 스캔 이동, ,/. : 시간 오프셋 변경

```bash
python tools/visualize_trajectory.py \
  --gps_csv data/test0820_23_36/GPS/Odom_data_pitch_flipped.csv \
  --lidar_dir data/test0820_23_36/lidar_xyzi \
  --lidar_index 10 \
  --extrinsics_yaml extrinsics.yaml \
  --target_rate 1~0 \
  --stride 5 \
  --x_range 20 --y_range 20 \
  --step_size 10 \
  --verbose
```

- **주요 옵션**
  - **--x_range, --y_range**: XY 크롭(절대값 반경)
  - **--heading_from_pose**: CSV 자세 대신 궤적에서 XY heading 계산

### 3) Point Cloud Merge — `tools/merge_lidar_gnss.py`
- **설명**: LiDAR 스캔들을 GNSS 시간축에 맞춰 **병합**하고, 궤적 마커와 함께 **인터랙티브**하게 시각화
- **입력**: `--gps_csv`, `--lidar_dir`, `--extrinsics_yaml`, `--time_offset`(나노초), 병합 범위 제어(`--start_index`, `--index_interval` 등)
- **키 조작**: ←/→ : 스캔 이동, ,/. : 시간 오프셋 변경

```bash
python tools/merge_lidar_gnss.py \
  --gps_csv data/test0820_23_36/GPS/Odom_data_pitch_flipped.csv \
  --lidar_dir data/test0820_23_36/lidar_xyzi \
  --time_offset 389000000 \
  --offset_step_ns 100000000 \
  --extrinsics_yaml extrinsics.yaml \
  --start_index 10 \
  --index_interval 10 \
  --max_frames 5 \
  --x_range 30 --y_range 30 --z_range 2 \
  --target_rate 10 \
  --verbose
```

- **유용한 옵션**
  - **--start_index / --index_interval**: 병합 시작 인덱스 / 간격(k)
  - **--max_frames N**: 한 번에 병합할 최대 스캔 수
  - **--max_points N**: 병합 포인트 수 상한(초과 시 트림)
  - **--marker_stride K**: 궤적 마커/화살표 간격
  - **--offset_step_ns**: ,/. 키당 시간 오프셋 스텝(ns)
