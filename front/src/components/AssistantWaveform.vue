<script setup>
import { ref, watch, onUnmounted } from 'vue';

const props = defineProps({
  userAnalyser: {
    type: Object, // AudioContext AnalyserNode
    default: null,
  },
  aiAnalyser: {
    type: Object, // AudioContext AnalyserNode
    default: null,
  },
  isVisualizing: {
    type: Boolean,
    default: false,
  },
  status: {
    type: String,
    required: true,
  }
});

const waveformCanvas = ref(null);
let animationFrameId = null;

// ⭐ 核心逻辑：使用 watch 监听 Analyser 节点的变化
watch(() => [props.userAnalyser, props.aiAnalyser, props.isVisualizing, props.status], () => {
  // 当任何一个依赖项变化时，这个函数都会重新执行
  cancelAnimationFrame(animationFrameId); // 先停止上一个动画

  if (!props.isVisualizing || !waveformCanvas.value) {
    // 如果不需要显示，或 canvas 还没准备好，就清空画布并返回
    const canvas = waveformCanvas.value;
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    return;
  }
  
  // 3. 基于 status
  if (props.status === 'speaking' && props.aiAnalyser) {
    visualize(props.aiAnalyser, true); // AI 说话，使用紫色
  } else if (props.status === 'listening' && props.userAnalyser) {
    visualize(props.userAnalyser, false); // 用户说话，使用白色
  }
}, { immediate: true }); // immediate: true 确保组件加载后立即执行一次


// 封装的可视化函数，现在接收 analyser 和 isAI 标志作为参数
function visualize(analyser, isAI) {
  if (!analyser || !waveformCanvas.value) return;
  const canvasCtx = waveformCanvas.value.getContext('2d');
  
  // --- 绘图逻辑 ---
  const barWidth = 6, spacing = 4, borderRadius = barWidth / 2, barSpacing = barWidth + spacing;
  const canvas = waveformCanvas.value, centerX = canvas.width / 2, centerY = canvas.height / 2;
  const maxBarHeight = canvas.height / 2.2;
  const bufferLength = analyser.frequencyBinCount, dataArray = new Uint8Array(bufferLength);
  const drawRoundedBar = (ctx, x, y, width, height) => { if (height <= 0) return; ctx.beginPath(); ctx.moveTo(x + borderRadius, y); ctx.lineTo(x + width - borderRadius, y); ctx.arcTo(x + width, y, x + width, y + borderRadius, borderRadius); ctx.lineTo(x + width, y + height); ctx.lineTo(x, y + height); ctx.lineTo(x, y + borderRadius); ctx.arcTo(x, y, x + borderRadius, y, borderRadius); ctx.closePath(); ctx.fill(); };
  const draw = () => { animationFrameId = requestAnimationFrame(draw); analyser.getByteFrequencyData(dataArray); canvasCtx.clearRect(0, 0, canvas.width, canvas.height); const barsToDraw = Math.floor((centerX - spacing) / barSpacing); for (let i = 0; i < barsToDraw; i++) { const dataIndex = Math.floor(i * (bufferLength / barsToDraw) * 0.9); const barHeight = (dataArray[dataIndex] / 255) * maxBarHeight; const reflectionHeight = barHeight * 0.4; const mainColor = isAI ? 'rgb(150, 120, 255)' : 'rgb(255, 255, 255)'; const reflectionColor = isAI ? 'rgba(150, 120, 255, 0.3)' : 'rgba(255, 255, 255, 0.3)'; const rightBarX = centerX + spacing / 2 + i * barSpacing; const leftBarX = centerX - spacing / 2 - barWidth - i * barSpacing; canvasCtx.fillStyle = mainColor; drawRoundedBar(canvasCtx, rightBarX, centerY - barHeight, barWidth, barHeight); canvasCtx.fillStyle = reflectionColor; drawRoundedBar(canvasCtx, rightBarX, centerY, barWidth, reflectionHeight); canvasCtx.fillStyle = mainColor; drawRoundedBar(canvasCtx, leftBarX, centerY - barHeight, barWidth, barHeight); canvasCtx.fillStyle = reflectionColor; drawRoundedBar(canvasCtx, leftBarX, centerY, barWidth, reflectionHeight); } };
  
  draw();
}

onUnmounted(() => {
  cancelAnimationFrame(animationFrameId);
});
</script>

<template>
  <div class="waveform-container">
    <canvas ref="waveformCanvas" width="600" height="100"></canvas>
  </div>
</template>

<style scoped>
.waveform-container { width: 100%; height: 100px; margin-top: 20px; display: flex; justify-content: center; align-items: center; }
canvas { width: 80%; max-width: 450px; height: 100%; }
</style>