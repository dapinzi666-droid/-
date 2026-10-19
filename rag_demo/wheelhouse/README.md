首次在用户 Windows 电脑实际下载对应 Python minor 的全部依赖 wheel。
按 win_amd64-cp311 或 win_amd64-cp312 和依赖配置指纹分开保存。
每个缓存含完整精确版本、文件 SHA256 和带哈希的 requirements-locked.txt。
恢复使用 --no-index --require-hashes，不拿 Linux 依赖缓存冒充 Windows 可用环境。
此轻量项目下载包不包含依赖二进制；完整本机备份才包含已下载的 wheel。
