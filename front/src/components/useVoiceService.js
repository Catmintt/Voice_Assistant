import { ref, onUnmounted } from 'vue';

/**
 * 这是一个组合式函数 (Composable)，封装了所有与实时语音助手服务相关的功能。
 * 它管理 WebSocket 连接、音频输入/输出处理以及会话状态。
 * 它是“无头”的，意味着它不依赖于任何特定的 UI 组件。
 */
export function useVoiceService() {
  // --- 1. 响应式状态 (Reactive State) ---
  // 这些状态将暴露给 Vue 组件，当它们改变时，UI 会自动更新。
  const status = ref('idle'); // 'idle' | 'connecting' | 'listening' | 'processing' | 'speaking'
  const finalAnswer = ref('');

  // 让 UI 组件自己决定如何根据这些节点进行可视化。
  const userAnalyser = ref(null);
  const aiAnalyser = ref(null);

  // --- 2. 内部状态变量 (Internal State Variables) ---
  // 这些是服务内部使用的变量，UI 组件不需要关心它们。
  let socket = null;
  let audioContext = null;
  let userSourceNode = null; // 麦克风的音频源
  let userWorkletNode = null; // 用于处理用户麦克风音频的 Worklet
  let mediaStream = null; // 用户的媒体流（麦克风）
  
  let ttsWorkletNode = null; // 用于播放 AI 语音的 Worklet
  let sttLocked = false; // 一个锁，当 AI 说话时为 true，防止此时采集用户语音
  let ttsPlaying = false; // 标记 AI 语音是否正在播放

  // --- 3. 核心功能函数 (内部逻辑) ---

  // 3.1: 设置 WebSocket 连接和消息处理
  function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const WS_URL = `ws://localhost:8000/ws/chat`; // TODO: 替换为后端地址
    
    console.log(`正在连接 WebSocket: ${WS_URL}`);
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      console.log("WebSocket 连接成功。");
      startListening(); // 连接成功后立即开始监听
    };

    socket.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      switch (message.type) {
        case 'stt_end':
          console.log("收到 STT 结束信号，停止监听。");
          stopListening();
          sttLocked = true;
          status.value = 'processing';
          break;
        
        case 'final_answer':
          finalAnswer.value = message.data;
          status.value = 'processing';
          await initTtsPlayer();
          sttLocked = true; 
          break;
        
        case 'tts_chunk':
          if (!sttLocked) {
            sttLocked = true;
            stopListening();
          }
          if (!ttsWorkletNode) { await initTtsPlayer(); }
          pushTtsChunkBase64(message.data);
          break;
        
        case 'tts_end':
          endTtsStream();
          break;
        
        case 'error':
          console.error("后端错误:", message.message);
          stopSession();
          break;
      }
    };

    socket.onerror = (error) => { console.error("WebSocket 错误:", error); stopSession(); };
    socket.onclose = () => { console.log("WebSocket 已关闭。"); status.value = 'idle'; };
  }

  // 3.2: 开始从麦克风采集、处理并发送音频
  async function startListening() {
    if (sttLocked || !socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    status.value = 'listening';
    finalAnswer.value = '';

    try {
      if (!mediaStream || !mediaStream.active) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        userSourceNode = audioContext.createMediaStreamSource(mediaStream);
        
        // 将 Analyser 节点暴露出去
        userAnalyser.value = audioContext.createAnalyser();
        userAnalyser.value.fftSize = 256;
        userSourceNode.connect(userAnalyser.value);

        // --- 创建用于重采样的 AudioWorklet ---
        const workletBlob = new Blob([`
        class ResamplerProcessor extends AudioWorkletProcessor {
          constructor(options) {
            super();
            this.targetSampleRate = options.processorOptions.targetSampleRate;
            this.buffer = [];
            this.bufferSize = 4096; // 缓冲区大小
          }

          process(inputs, outputs, parameters) {
            const input = inputs[0][0]; // 获取第一个通道的数据
            if (input) {
              const resampled = this.resample(input, sampleRate, this.targetSampleRate);
              const int16Pcm = this.float32ToInt16(resampled);
              this.port.postMessage(int16Pcm.buffer, [int16Pcm.buffer]);
            }
            return true;
          }

          float32ToInt16(buffer) {
            let l = buffer.length;
            const buf = new Int16Array(l);
            while (l--) {
              buf[l] = Math.min(1, buffer[l]) * 0x7FFF;
            }
            return buf;
          }

          resample(audioBuffer, fromSampleRate, toSampleRate) {
            if (fromSampleRate === toSampleRate) {
              return audioBuffer;
            }
            const ratio = toSampleRate / fromSampleRate;
            const newLength = Math.round(audioBuffer.length * ratio);
            const result = new Float32Array(newLength);
            let lastPos = 0, newPos = 0;
            for (let i = 0; i < newLength; i++) {
              const pos = i / ratio;
              const low = Math.floor(pos);
              const high = Math.ceil(pos);
              const lv = audioBuffer[low] || 0;
              const hv = audioBuffer[high] || 0;
              result[i] = lv + (hv - lv) * (pos - low);
            }
            return result;
          }
        }
        registerProcessor('resampler-processor', ResamplerProcessor);
      `], { type: 'application/javascript' });
        
        const workletURL = URL.createObjectURL(workletBlob);
        await audioContext.audioWorklet.addModule(workletURL);

        userWorkletNode = new AudioWorkletNode(audioContext, 'resampler-processor', {
          processorOptions: { targetSampleRate: 16000 }
        });

        userWorkletNode.port.onmessage = (event) => {
          if (sttLocked || !socket || socket.readyState !== WebSocket.OPEN) return;
          const u8 = new Uint8Array(event.data);
          const b64 = btoa(String.fromCharCode.apply(null, u8));
          socket.send(b64);
        };
        
        userSourceNode.connect(userWorkletNode);
      }

      mediaStream.getTracks().forEach(track => track.enabled = true);
      console.log("麦克风已激活并开始监听。");

    } catch (err) {
      console.error("无法获取麦克风权限或设置音频处理:", err);
      alert("请允许使用麦克风权限。");
      stopSession();
    }
  }
  
  // 3.3: 停止用户语音采集（静音）
  function stopListening() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.enabled = false);
      console.log("麦克风已静音。");
    }
  }

  // 3.4: 初始化 TTS 音频播放器
  async function initTtsPlayer() {
    if (!ttsWorkletNode) {
      const ttsPlayerBlob = new Blob([`
      class PCMPlayer extends AudioWorkletProcessor {
        constructor(options) {
          super();
          this.inputRate = (options.processorOptions && options.processorOptions.inputSampleRate) || 24000;
          this.buffer = new Float32Array(0);
          this.ended = false;
          this.port.onmessage = (e) => {
            const msg = e.data || {};
            if (msg.type === 'chunk') {
              // msg.data 是 Int16Array（单声道）
              const i16 = new Int16Array(msg.data);
              const f32 = new Float32Array(i16.length);
              for (let i = 0; i < i16.length; i++) {
                f32[i] = Math.max(-1, Math.min(1, i16[i] / 32768));
              }
              const res = this.resampleLinear(f32, this.inputRate, sampleRate);
              this.enqueue(res);
            } else if (msg.type === 'end') {
              this.ended = true;
            }
          };
        }

        enqueue(f32) {
          const merged = new Float32Array(this.buffer.length + f32.length);
          merged.set(this.buffer, 0);
          merged.set(f32, this.buffer.length);
          this.buffer = merged;
        }

        resampleLinear(input, from, to) {
          if (from === to) return input;
          const ratio = to / from;
          const outLen = Math.floor(input.length * ratio);
          const out = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            const pos = i / ratio;
            const idx = Math.floor(pos);
            const frac = pos - idx;
            const s0 = input[idx] || 0;
            const s1 = input[idx + 1] || s0;
            out[i] = s0 + (s1 - s0) * frac;
          }
          return out;
        }

        process(inputs, outputs) {
          // 单声道输出：outputs[0][0]
          const out = outputs[0][0];
          const n = out.length;

          if (this.buffer.length >= n) {
            out.set(this.buffer.subarray(0, n));
            this.buffer = this.buffer.subarray(n);
          } else {
            // 不足时用 0 填充，避免杂音
            out.set(this.buffer);
            for (let i = this.buffer.length; i < n; i++) out[i] = 0;
            this.buffer = new Float32Array(0);

            if (this.ended) {
              this.port.postMessage({ type: 'ended' });
              // 立刻重置标志位，防止重复发送！
              this.ended = false;
            }
          }
          return true; // 持续运行
        }
      }
      registerProcessor('pcm-player', PCMPlayer);
    `], { type: 'application/javascript' });
      
      const url = URL.createObjectURL(ttsPlayerBlob);
      await audioContext.audioWorklet.addModule(url);
      ttsWorkletNode = new AudioWorkletNode(audioContext, 'pcm-player', { processorOptions: { inputSampleRate: 24000 } });
      
      // 创建并暴露 AI 的 Analyser 节点
      aiAnalyser.value = audioContext.createAnalyser();
      aiAnalyser.value.fftSize = 256;

      ttsWorkletNode.connect(aiAnalyser.value);
      aiAnalyser.value.connect(audioContext.destination);

      ttsWorkletNode.port.onmessage = (e) => {
        if (e.data?.type === 'ended') {
          ttsPlaying = false;
          sttLocked = false;
          if (socket && socket.readyState === WebSocket.OPEN) {
            startListening(); // AI 说完，自动开始听用户说
          } else {
            stopSession();
          }
        }
      };
    }
  }

  // 3.5: 接收并处理 TTS 音频块
  function pushTtsChunkBase64(b64) {
    if (!ttsWorkletNode) return;
    const bin = atob(b64);
    const len = bin.length;
    const i16 = new Int16Array(len / 2);
    for (let i = 0; i < len; i += 2) {
      const lo = bin.charCodeAt(i);
      const hi = bin.charCodeAt(i + 1);
      i16[i / 2] = (hi << 8) | lo;
    }
    ttsWorkletNode.port.postMessage({ type: 'chunk', data: i16.buffer }, [i16.buffer]);

    if (!ttsPlaying) {
      status.value = 'speaking';
      ttsPlaying = true;
    }
  }

  // 3.6: 标记 TTS 音频流结束
  function endTtsStream() {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: 'end' });
    }
  }

  // --- 4. 公开的控制方法 (Public Control Methods) ---
  // 这些是暴露给 Vue 组件的“遥控器”按钮。
  function startSession() {
    if (status.value !== 'idle') return;
    status.value = 'connecting';
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    setupWebSocket();
  }

  function stopSession() {
    if (status.value === 'idle') return;
    
    stopListening();
    if (socket) { socket.close(); socket = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; }
    if (userSourceNode) { userSourceNode.disconnect(); userSourceNode = null; }
    if (userWorkletNode) { userWorkletNode.disconnect(); userWorkletNode = null; }
    if (ttsWorkletNode) { ttsWorkletNode.disconnect(); ttsWorkletNode = null; }
    
    // 重置所有状态
    sttLocked = false;
    ttsPlaying = false;
    status.value = 'idle';
    userAnalyser.value = null;
    aiAnalyser.value = null;
  }
  
  function toggleSession() {
      if (status.value === 'idle') {
          startSession();
      } else {
          stopSession();
      }
  }

  // --- 5. 生命周期钩子 (Lifecycle Hook) ---
  // 确保当使用此 Composable 的组件被销毁时，所有资源都被正确释放。
  onUnmounted(() => {
    stopSession();
    if (audioContext) { audioContext.close(); audioContext = null; }
  });

  // --- 6. 返回值 (The Return Value) ---
  // 将需要暴露给组件的状态和方法在这里返回。
  return {
    // 响应式状态
    status,
    finalAnswer,
    // 可视化所需的数据源
    userAnalyser,
    aiAnalyser,
    // 控制函数
    toggleSession,
  };
}