<script setup>
import { computed } from 'vue';
import { useVoiceService } from './useVoiceService';
import AssistantAvatar from './AssistantAvatar.vue';
import AssistantStatus from './AssistantStatus.vue';
import AssistantControls from './AssistantControls.vue';
import AssistantWaveform from './AssistantWaveform.vue';

const { status, userAnalyser, aiAnalyser, toggleSession } = useVoiceService();

// --- 计算属性，用于驱动 UI ---
const isAiSpeaking = computed(() => status.value === 'speaking');
const isUserListening = computed(() => status.value === 'listening');
const isVisualizing = computed(() => isAiSpeaking.value || isUserListening.value);

const statusText = computed(() => {
  // 当 AI 说话且有最终答案时，可以考虑显示答案内容（此为可选交互）
//   if (isAiSpeaking.value && finalAnswer.value) {
//     return finalAnswer.value;
//   }
  switch (status.value) {
    case 'connecting': return '正在连接...';
    case 'listening': return '我在听，请讲...';
    case 'processing': return '我正在思考哦...';
    case 'speaking': return '请仔细听我的回复哦...';
    default: return '请点击下方麦克风开始对话哦';
  }
});

const statusSubtitle = computed(() => {
  return status.value === 'idle' ? '对话内容均为AI生成' : '';
});

// --- 事件处理 ---
function handleToggleSession() {
  // 直接调用从 Composable 中获取的方法
  toggleSession();
}
</script>

<template>
  <div class="assistant-container">
    <main class="main-content">
      <AssistantAvatar :is-speaking="isAiSpeaking" />
      <AssistantStatus :title="statusText" :subtitle="statusSubtitle" />
      
      <!-- 将 Analyser 节点和显隐状态传递给波形图组件 -->
      <AssistantWaveform
        :user-analyser="userAnalyser"
        :ai-analyser="aiAnalyser"
        :is-visualizing="isVisualizing"
        :status="status"
      />
    </main>

    <AssistantControls :status="status" @toggle-session="handleToggleSession" />
  </div>
</template>

<style scoped>
.assistant-container { width: 100vw; height: 100vh; background: radial-gradient(circle at 50% 30%, #4a3b8b, #1a1a2e 70%); display: flex; flex-direction: column; justify-content: center; color: white; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }
.main-content { flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-top: -50px; }
.waveform-container { width: 100%; height: 100px; margin-top: 20px; display: flex; justify-content: center; align-items: center; }
</style>