<script setup>
defineProps({
  isSpeaking: {
    type: Boolean,
    default: false,
  },
});
</script>

<template>
  <div class="avatar-wrapper" :class="{ 'is-speaking': isSpeaking }">
    <div class="halo halo-1"></div>
    <div class="halo halo-2"></div>
    <img src="/images/avatar.png" alt="AI Assistant Avatar" class="avatar-image" />
  </div>
</template>

<style scoped>
.avatar-wrapper { position: relative; width: 200px; height: 200px; display: flex; justify-content: center; align-items: center; margin-top: -50px; }
.avatar-image { width: 100%; height: 100%; border-radius: 50%; object-fit: cover; z-index: 2; position: relative; }
.halo { position: absolute; border-radius: 50%; border: 2px solid rgba(150, 120, 255, 0); width: 100%; height: 100%; transition: all 0.3s ease; }
.avatar-wrapper.is-speaking .halo-1 { animation: halo-pulse 2s infinite ease-in-out; }
.avatar-wrapper.is-speaking .halo-2 { animation: halo-pulse 2s infinite ease-in-out 0.6s; }
@keyframes halo-pulse { 0% { transform: scale(1); border-color: rgba(150, 120, 255, 0.8); border-width: 3px; } 50% { transform: scale(1.3); border-color: rgba(150, 120, 255, 0.4); border-width: 2px; } 100% { transform: scale(1.6); border-color: rgba(150, 120, 255, 0); border-width: 1px; } }
.avatar-wrapper.is-speaking .avatar-image { box-shadow: 0 0 20px rgba(150, 120, 255, 0.6), 0 0 40px rgba(150, 120, 255, 0.4), 0 0 60px rgba(150, 120, 255, 0.2); animation: glow-pulse 2s infinite ease-in-out; }
@keyframes glow-pulse { 0%, 100% { box-shadow: 0 0 20px rgba(150, 120, 255, 0.6), 0 0 40px rgba(150, 120, 255, 0.4), 0 0 60px rgba(150, 120, 255, 0.2); } 50% { box-shadow: 0 0 30px rgba(150, 120, 255, 0.8), 0 0 60px rgba(150, 120, 255, 0.6), 0 0 90px rgba(150, 120, 255, 0.4); } }
</style>