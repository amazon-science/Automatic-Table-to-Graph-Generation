# AutoG-S Web App - Caching Strategy

## Overview

The AutoG-S web app uses an intelligent hybrid caching strategy to optimize performance and memory usage for intermediate files generated during processing.

## File Categories

### 📝 Small Text Files (Memory Cached)
- **`information.txt`** - Data analysis results (1-10KB)
- **`metadata.yaml`** - Dataset metadata (1-10KB) 
- **`agent_history.txt`** - Processing history (1-50KB)
- **`prompt.txt`** - Debug prompts (1-5KB)

**Strategy:** Always cached in memory for instant access and downloads.

### 🗄️ Large Binary Files (Disk Only)
- **`deepjoin.pkl`** - Join discovery cache (1-100MB)
- **`backup_*`** - State backups (1-50MB each)
- **`*.parquet`** - Data tables (varies)
- **`*.npy/.npz`** - NumPy arrays (varies)

**Strategy:** Stored on disk only to preserve memory.

### 🖼️ Media Files (Conditional)
- **`schema.png`** - Schema diagrams (<1MB → memory, >1MB → disk)
- **`schema.pdf`** - Schema PDFs (varies)

**Strategy:** Small images cached in memory, large files on disk.

## Cache Strategies

### 🔄 Hybrid (Recommended)
- **Small text files** → Memory cache
- **Large binary files** → Disk storage
- **Images** → Memory if <1MB, disk otherwise
- **Memory limit:** 10MB total

### 💾 Memory
- **All files** → Memory cache
- **Best for:** Small datasets, fast access
- **Risk:** High memory usage

### 💿 Disk
- **All files** → Disk storage
- **Best for:** Large datasets, memory-constrained environments
- **Trade-off:** Slower file access

## Performance Benefits

### Memory Cache Advantages
- ⚡ **Instant Downloads** - No disk I/O for cached files
- 🔄 **Multiple Access** - Files can be downloaded repeatedly without re-reading
- 📊 **Better UX** - Faster response times for small files

### Disk Storage Advantages
- 💾 **Memory Efficiency** - Large files don't consume RAM
- 🔄 **Persistence** - Files survive between operations
- 📈 **Scalability** - Handles datasets of any size

## Implementation Details

### Automatic File Classification
```python
def _should_cache_in_memory(self, file_path: str, content_size: int = None) -> bool:
    # Text files < 10MB → Memory
    # Binary files → Disk
    # Images < 1MB → Memory
```

### Cache Management
- **Size Monitoring** - Tracks total memory usage
- **Automatic Cleanup** - Clears cache on session end
- **Fallback Logic** - Disk access if memory cache fails

### User Control
- **Strategy Selection** - Choose hybrid/memory/disk in UI
- **Cache Information** - View cached files and memory usage
- **Real-time Updates** - Strategy changes apply immediately

## Best Practices

### For Small Datasets (<100MB)
- Use **Memory** strategy for maximum speed
- All files fit comfortably in RAM

### For Medium Datasets (100MB-1GB)
- Use **Hybrid** strategy (default)
- Optimal balance of speed and memory usage

### For Large Datasets (>1GB)
- Use **Disk** strategy to prevent memory issues
- Prioritize stability over speed

### For Memory-Constrained Environments
- Always use **Disk** strategy
- Monitor system memory usage

## Monitoring

### Cache Information Display
- **Files Cached:** Count and list of memory-cached files
- **Memory Usage:** Total MB used by cache
- **Strategy:** Current caching approach
- **File Sizes:** Individual file size reporting

### Performance Metrics
- **Cache Hit Rate:** Files served from memory vs disk
- **Download Speed:** Time to serve cached vs disk files
- **Memory Efficiency:** Cache size vs total file size

## Technical Implementation

### Cache Storage
```python
self.memory_cache = {}  # Dict[file_path, bytes]
self.cache_size_limit = 10 * 1024 * 1024  # 10MB
```

### File Access Pattern
1. **Check Memory Cache** - Instant return if available
2. **Read from Disk** - Load file if not cached
3. **Cache Decision** - Store in memory if appropriate
4. **Return Content** - Serve to user

### Cleanup Process
- **Session End** - Clear all memory cache
- **Temp Directory** - Remove all disk files
- **Error Handling** - Graceful degradation if cache fails

This intelligent caching system ensures optimal performance while maintaining memory efficiency across different dataset sizes and system configurations.