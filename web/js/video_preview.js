import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

console.log("[SubtitleDetector] Loading video preview extension...");

// 为 VideoCombine 节点添加视频预览功能
app.registerExtension({
    name: "SubtitleDetector.VideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[SubtitleDetector] Checking node:", nodeData.name);
        if (nodeData.name === "VideoCombine") {
            console.log("[SubtitleDetector] Registering VideoCombine preview...");
            // 在节点创建时添加预览 widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const previewNode = this;

                // 创建预览容器
                const element = document.createElement("div");

                // 添加 DOM widget
                const previewWidget = this.addDOMWidget("videopreview", "preview", element, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() {
                        return element.value;
                    },
                    setValue(v) {
                        element.value = v;
                    },
                });

                // 计算 widget 大小
                previewWidget.computeSize = function(width) {
                    if (this.aspectRatio && !this.parentEl.hidden) {
                        let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                        if (!(height > 0)) {
                            height = 0;
                        }
                        return [width, height];
                    }
                    return [width, -4]; // 没有加载源时，widget 不显示
                };

                // 初始化值
                previewWidget.value = { hidden: false, paused: false, params: {} };

                // 创建父容器
                previewWidget.parentEl = document.createElement("div");
                previewWidget.parentEl.className = "vhs_preview";
                previewWidget.parentEl.style.width = "100%";
                element.appendChild(previewWidget.parentEl);

                // 创建视频元素
                previewWidget.videoEl = document.createElement("video");
                previewWidget.videoEl.controls = true;
                previewWidget.videoEl.loop = true;
                previewWidget.videoEl.muted = true;
                previewWidget.videoEl.style.width = "100%";

                // 视频加载完成时设置宽高比
                previewWidget.videoEl.addEventListener("loadedmetadata", () => {
                    previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
                    previewNode.setSize([
                        previewNode.size[0],
                        previewNode.computeSize()[1]
                    ]);
                });

                // 视频错误时隐藏
                previewWidget.videoEl.addEventListener("error", () => {
                    previewWidget.parentEl.hidden = true;
                });

                // 鼠标悬停时取消静音
                previewWidget.videoEl.onmouseenter = () => {
                    previewWidget.videoEl.muted = false;
                };
                previewWidget.videoEl.onmouseleave = () => {
                    previewWidget.videoEl.muted = true;
                };

                previewWidget.parentEl.appendChild(previewWidget.videoEl);

                // 更新预览源的函数
                previewWidget.updateSource = function() {
                    if (!this.value.params || !this.value.params.filename) {
                        return;
                    }

                    const params = Object.assign({}, this.value.params);
                    params.timestamp = Date.now();

                    this.parentEl.hidden = this.value.hidden;

                    if (params.format?.split('/')[0] === 'video') {
                        this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                        this.videoEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                        this.videoEl.hidden = false;
                    }
                };

                // 更新参数的函数
                this.updateParameters = (params) => {
                    if (!previewWidget.value.params) {
                        previewWidget.value.params = {};
                    }
                    Object.assign(previewWidget.value.params, params);
                    previewWidget.updateSource();
                };

                return result;
            };

            // 在节点执行后更新预览
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                console.log("[SubtitleDetector] onExecuted called, message:", message);
                const result = onExecuted?.apply(this, arguments);

                if (message && message.gifs && message.gifs.length > 0) {
                    console.log("[SubtitleDetector] Updating preview with:", message.gifs[0]);
                    this.updateParameters(message.gifs[0]);
                } else {
                    console.log("[SubtitleDetector] No gifs data in message");
                }

                return result;
            };
        }
    },
});
