# EV-SVC — Event-based Drone Detection (Learning-free v2)

이벤트 카메라 스트림에서 학습 없이(learning-free) 드론을 탐지하는 파이프라인입니다.
33ms 윈도우 단위로 EMA 배경 제거 → residual event 추출 → (x, y, t) voxel 3D 연결요소 분석을 통해
바운딩 박스를 추출하고, 간단한 centroid 기반 tracklet 매칭으로 추적합니다.

## 구성 파일

| 파일 | 설명 |
|---|---|
| `detect_v2.py` | 탐지 파이프라인 메인 스크립트 |
| `detectv2.yaml` | 실행 파라미터(config) |

## Pipeline

| Stage | 내용 |
|---|---|
| 0 | EMA 기반 배경(baseline) 추정 |
| 1 | Residual event 추출 (배경 대비 활성 셀만) |
| 2 | (x, y, t) voxel grid 구성 (5px × 1ms) |
| 3 | 3D Connected Component (6-connectivity) |
| 4 | Component feature 필터링 (duration / displacement / size) → confirmed bbox |

이후 centroid distance 기반 greedy matching으로 tracklet을 구성하고, `age_min` 이상 유지된 blob만 최종 출력합니다.

## 실행 방법

python3 detect_v2.py --config detectv2.yaml --seqs 81 82
python3 detect_v2.py --config detectv2.yaml --seq 82 --vis --max-windows 200

option
--config PATH	YAML 파일 지정
--seq N	시퀀스 하나만 실행
--seqs N N ...	복수 시퀀스 실행
--vis	평가 + PNG 저장 (vis_interval마다)
--vis-only	PNG 저장만, 평가/CSV 없음
--vod	mp4 저장, 평가 없음
--rgb	--vod와 함께: RGB+이벤트 합성 영상
--max-windows N	처음 N 윈도우만 실행 (디버깅용)
--save-video	평가와 동시에 mp4 저장
--null-test	GT bbox 랜덤 셔플 → null test



## 영상 결과
 https://www.youtube.com/watch?v=7NSE5ZO_3hc
야간 https://www.youtube.com/watch?v=zT8xIZ-AG9U
실내 https://www.youtube.com/watch?v=LnpyTfbSeaA
비 https://www.youtube.com/watch?v=YAYTyZet81o
먼거리 https://www.youtube.com/watch?v=nXzRNFcgMkg
