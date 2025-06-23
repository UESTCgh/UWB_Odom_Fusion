#!/bin/bash
echo "🚀 开始使用 evo 评估并可视化所有轨迹..."

TRAJ_DIR="./trajectories"
OUT_DIR="$TRAJ_DIR/eval_results"
mkdir -p "$OUT_DIR"

for ID in 0 1 2 3
do
  UWB="$TRAJ_DIR/id${ID}_uwb.tum"
  ICP="$TRAJ_DIR/id${ID}_odom.tum"
  PLOT_APE="$OUT_DIR/id${ID}_ape_plot.png"
  PLOT_TRAJ="$OUT_DIR/id${ID}_traj_plot.png"
  LOG="$OUT_DIR/id${ID}_eval.txt"

  if [[ -f "$UWB" && -f "$ICP" ]]; then
    echo "📈 正在评估 ID=$ID..."

    # APE 评估（只使用 x,y）
    evo_ape tum "$UWB" "$ICP" --align -va --save_plot "$PLOT_APE" > "$LOG"

    # 轨迹可视化（只使用 x,y）
    evo_traj tum "$UWB" "$ICP" --ref="$UWB" --save_plot "$PLOT_TRAJ"

  else
    echo "⚠️ 缺失轨迹文件: ID=$ID"
  fi
done

echo "✅ 所有轨迹评估与可视化完成，结果保存在 $OUT_DIR"
