"""
GPU-accelerated batch prediction for DESI DLA finder
Optimized for window-level batching
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import sys

sys.path.append('/global/cfs/cdirs/desi/users/jqzou')

from desidlas.datasets.get_flux import make_dataset
from desidlas.parameters import kernel
from desidlas.training.parameterset import parameter_names, parameters


class Timer:
    """计时器工具类"""
    
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
    
    def format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}min"
        else:
            return f"{seconds/3600:.2f}h"
    
    def __str__(self):
        return self.format_time(self.elapsed)


class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    def record(self, name, elapsed, count=1):
        """记录时间"""
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        self.timings[name].append(elapsed)
        self.counts[name] += count
    
    def get_stats(self, name):
        """获取统计信息"""
        if name not in self.timings:
            return None
        times = self.timings[name]
        count = self.counts[name]
        return {
            'total': sum(times),
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': count,
            'throughput': count / sum(times) if sum(times) > 0 else 0
        }
    
    def print_summary(self):
        """打印汇总统计"""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Total time:    {Timer('').format_time(stats['total'])}")
            print(f"  Average time:  {Timer('').format_time(stats['mean'])}")
            print(f"  Min/Max:       {Timer('').format_time(stats['min'])} / {Timer('').format_time(stats['max'])}")
            print(f"  Count:         {stats['count']}")
            if 'window' in name.lower() or 'sightline' in name.lower():
                print(f"  Throughput:    {stats['throughput']:.2f} items/sec")
        
        print("\n" + "="*80)


class DLAModelGPU:
    """GPU版本的DLA检测模型包装器"""
    
    def __init__(self):
        """初始化并加载三个模型（high/mid/low SNR）"""
        self.models = {}
        self.model_paths = {
            'high': '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_highsnr/train_highsnr/current_99999',
            'mid': '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_midsnr/train_midsnr/current_99999',
            'low': '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_lowsnr/train_lowsnr/current_99999'
        }
        
        print(f"\n{'='*80}")
        print("LOADING DLA MODELS")
        print(f"{'='*80}")
        
        total_load_time = 0
        for model_type in ['high', 'mid', 'low']:
            print(f"\nLoading {model_type.upper()} SNR model...")
            with Timer(f"{model_type}_model_load") as t:
                self.models[model_type] = self._load_tf1_model(
                    self.model_paths[model_type], 
                    model_type
                )
            total_load_time += t.elapsed
            print(f"  ✓ Loaded in {t}")
        
        print(f"\n{'='*80}")
        print(f"All models loaded in {Timer('').format_time(total_load_time)}")
        print(f"{'='*80}\n")
    
    def _load_tf1_model(self, checkpoint_path, model_type):
        """从TF1的checkpoint加载模型到TF2"""
        h5_path = checkpoint_path.replace('current_99999', 'model.h5')
        if os.path.exists(h5_path):
            return keras.models.load_model(h5_path, compile=False)
        
        print(f"    Using TF1 checkpoint format")
        
        class TF1ModelWrapper:
            def __init__(self, ckpt_path):
                self.ckpt_path = ckpt_path
                self.graph = None
                self.sess = None
                self._load()
            
            def _load(self):
                """加载TF1模型，自动处理设备映射"""
                self.graph = tf.compat.v1.Graph()
                with self.graph.as_default():
                    config = tf.compat.v1.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.allow_soft_placement = True
                    config.log_device_placement = False
                    
                    self.sess = tf.compat.v1.Session(config=config)
                    
                    saver = tf.compat.v1.train.import_meta_graph(
                        f'{self.ckpt_path}.ckpt.meta',
                        clear_devices=True
                    )
                    
                    saver.restore(self.sess, f'{self.ckpt_path}.ckpt')
                    print(f"      Model loaded and remapped to available GPU")
            
            def predict(self, flux_batch):
                """批量预测"""
                with self.graph.as_default():
                    pred = self.graph.get_tensor_by_name('prediction:0')
                    conf = self.graph.get_tensor_by_name('output_classifier:0')
                    offset = self.graph.get_tensor_by_name('y_nn_offset:0')
                    coldensity = self.graph.get_tensor_by_name('y_nn_coldensity:0')
                    x = self.graph.get_tensor_by_name('x:0')
                    keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
                    
                    results = self.sess.run(
                        [pred, conf, offset, coldensity],
                        feed_dict={x: flux_batch, keep_prob: 1.0}
                    )
                return results
            
            def __del__(self):
                if self.sess is not None:
                    self.sess.close()
        
        return TF1ModelWrapper(checkpoint_path)
    
    def predict_batch(self, flux_batch, model_type='mid'):
        """批量预测"""
        model = self.models[model_type]
        
        if isinstance(model, keras.Model):
            predictions = model.predict(flux_batch, batch_size=len(flux_batch), verbose=0)
            return predictions
        else:
            return model.predict(flux_batch)


def pred_sightline_batch_gpu(sightlines_batch, model_gpu, monitor, max_windows_per_batch=4096):
    """
    批量处理多条视线 - 窗口级批处理优化版
    """
    results = []
    
    # 第一步：收集所有数据并按SNR分组
    with Timer("data_preprocessing") as t:
        all_flux = {'high': [], 'mid': [], 'low': []}
        all_metadata = {'high': [], 'mid': [], 'low': []}
        
        for idx, sightline in enumerate(sightlines_batch):
            if sightline == [] or sightline is None:
                results.append(None)
                continue
            
            try:
                flux, lam = make_dataset(sightline)
                
                # 修正lam形状
                if len(lam.shape) == 2:
                    lam = lam[0]
                
                n_windows = flux.shape[0]
                
                # 根据SNR分类
                if hasattr(sightline, 's2n'):
                    if sightline.s2n >= 7:
                        model_type = 'high'
                    elif sightline.s2n >= 3:
                        model_type = 'mid'
                    else:
                        model_type = 'low'
                else:
                    model_type = 'mid'
                
                all_flux[model_type].append(flux)
                all_metadata[model_type].append({
                    'sightline_idx': idx,
                    'n_windows': n_windows,
                    'lam': lam
                })
                
            except Exception as e:
                print(f"    Warning: Error preprocessing sightline {idx}: {e}")
                results.append(None)
    
    monitor.record('preprocessing', t.elapsed, len(sightlines_batch))
    
    # 第二步：对每个SNR类别进行大batch预测
    all_predictions = {}
    
    for model_type in ['high', 'mid', 'low']:
        if len(all_flux[model_type]) == 0:
            continue
        
        # 合并所有flux
        flux_combined = np.concatenate(all_flux[model_type], axis=0)
        total_windows = flux_combined.shape[0]
        metadata_list = all_metadata[model_type]
        
        print(f"    {model_type.upper()} SNR: {len(metadata_list)} sightlines, {total_windows} windows")
        
        # 第三步：分批GPU推理
        all_pred = []
        all_conf = []
        all_offset = []
        all_coldensity = []
        
        num_window_batches = (total_windows + max_windows_per_batch - 1) // max_windows_per_batch
        
        with Timer(f"gpu_inference_{model_type}") as t:
            for batch_idx in range(num_window_batches):
                start_idx = batch_idx * max_windows_per_batch
                end_idx = min(start_idx + max_windows_per_batch, total_windows)
                
                window_batch = flux_combined[start_idx:end_idx]
                
                pred, conf, offset, coldensity = model_gpu.predict_batch(window_batch, model_type)
                
                all_pred.append(pred)
                all_conf.append(conf)
                all_offset.append(offset)
                all_coldensity.append(coldensity)
        
        # 合并结果
        all_pred = np.concatenate(all_pred, axis=0)
        all_conf = np.concatenate(all_conf, axis=0)
        all_offset = np.concatenate(all_offset, axis=0)
        all_coldensity = np.concatenate(all_coldensity, axis=0)
        
        monitor.record(f'inference_{model_type}', t.elapsed, total_windows)
        
        # 第四步：分配回各个sightline
        window_idx = 0
        for metadata in metadata_list:
            sightline_idx = metadata['sightline_idx']
            n_windows = metadata['n_windows']
            lam = metadata['lam']
            
            all_predictions[sightline_idx] = {
                'pred': all_pred[window_idx:window_idx + n_windows],
                'conf': all_conf[window_idx:window_idx + n_windows],
                'offset': all_offset[window_idx:window_idx + n_windows],
                'coldensity': all_coldensity[window_idx:window_idx + n_windows],
                'lam': lam
            }
            
            window_idx += n_windows
    
    # 第五步：按原始顺序返回
    final_results = []
    for idx in range(len(sightlines_batch)):
        if idx in all_predictions:
            final_results.append(all_predictions[idx])
        else:
            final_results.append(None)
    
    return final_results


def predictions_desi_gpu(pred_sightlines, savefile, batch_size=128, max_windows_per_batch=4096):
    """GPU批处理主函数"""
    
    monitor = PerformanceMonitor()
    overall_start = time.time()
    
    print("\n" + "="*80)
    print("GPU BATCH PREDICTION FOR DESI DLA FINDER")
    print("="*80)
    print(f"Start time:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total files:             {len(pred_sightlines)}")
    print(f"Sightlines per batch:    {batch_size}")
    print(f"Windows per GPU batch:   {max_windows_per_batch}")
    print("="*80 + "\n")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠ WARNING: No GPU detected!")
    else:
        print(f"✓ Using {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    print()
    
    with Timer("model_loading") as t:
        model_gpu = DLAModelGPU()
    monitor.record('model_loading', t.elapsed)
    
    for file_idx, (input_file, output_file) in enumerate(zip(pred_sightlines, savefile)):
        file_start = time.time()
        
        print("="*80)
        print(f"FILE {file_idx+1}/{len(pred_sightlines)}")
        print("="*80)
        print(f"Input:  {os.path.basename(input_file)}")
        print(f"Output: {os.path.basename(output_file)}")
        print(f"Time:   {datetime.now().strftime('%H:%M:%S')}")
        print("-"*80)
        
        with Timer("file_loading") as t:
            try:
                sightlines_data = np.load(input_file, allow_pickle=True, encoding='latin1')
                sightlines_array = sightlines_data.ravel()
                total_sightlines = len(sightlines_array)
            except Exception as e:
                print(f"✗ ERROR loading file: {e}\n")
                continue
        
        print(f"Loaded {total_sightlines} sightlines in {t}")
        monitor.record('file_loading', t.elapsed)
        
        all_results = []
        num_batches = (total_sightlines + batch_size - 1) // batch_size
        
        print(f"Processing in {num_batches} sightline batches...")
        
        batch_times = []
        prediction_start = time.time()
        
        pbar = tqdm(range(num_batches), 
                   desc="  Progress",
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch_idx in pbar:
            batch_start_time = time.time()
            
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_sightlines)
            
            sightlines_batch = sightlines_array[batch_start:batch_end]
            
            try:
                batch_results = pred_sightline_batch_gpu(
                    sightlines_batch, 
                    model_gpu, 
                    monitor,
                    max_windows_per_batch=max_windows_per_batch
                )
                all_results.extend(batch_results)
                
                batch_elapsed = time.time() - batch_start_time
                batch_times.append(batch_elapsed)
                
                if len(batch_times) > 0:
                    avg_batch_time = np.mean(batch_times[-5:])
                    remaining_batches = num_batches - batch_idx - 1
                    eta_seconds = avg_batch_time * remaining_batches
                    
                    pbar.set_postfix({
                        'time': f'{batch_elapsed:.1f}s',
                        'ETA': Timer('').format_time(eta_seconds)
                    })
                
            except Exception as e:
                print(f"\n✗ ERROR in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                all_results.extend([None] * len(sightlines_batch))
        
        pbar.close()
        
        prediction_elapsed = time.time() - prediction_start
        file_elapsed = time.time() - file_start
        
        print("-"*80)
        print("FILE STATISTICS:")
        print(f"  Total sightlines:     {total_sightlines}")
        print(f"  Prediction time:      {Timer('').format_time(prediction_elapsed)}")
        print(f"  File total time:      {Timer('').format_time(file_elapsed)}")
        print(f"  Throughput:           {total_sightlines/prediction_elapsed:.2f} sightlines/sec")
        print(f"  Avg per sightline:    {prediction_elapsed/total_sightlines*1000:.2f} ms")
        if len(batch_times) > 0:
            print(f"  Avg per batch:        {np.mean(batch_times):.3f}s")
        
        monitor.record('file_prediction', prediction_elapsed, total_sightlines)
        monitor.record('file_total', file_elapsed, total_sightlines)
        
        with Timer("file_saving") as t:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.save(output_file, np.array(all_results, dtype=object))
                print(f"  Save time:            {t}")
                print(f"✓ Saved to: {os.path.basename(output_file)}")
            except Exception as e:
                print(f"✗ ERROR saving file: {e}")
        
        monitor.record('file_saving', t.elapsed)
        print()
    
    overall_elapsed = time.time() - overall_start
    
    print("="*80)
    print("ALL FILES COMPLETED!")
    print("="*80)
    print(f"End time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime:     {Timer('').format_time(overall_elapsed)}")
    print(f"Files processed:   {len(pred_sightlines)}")
    print(f"Avg per file:      {Timer('').format_time(overall_elapsed/len(pred_sightlines))}")
    print("="*80)
    
    monitor.print_summary()


def predictions_desi(pred_sightlines, savefile):
    """兼容旧接口"""
    print("INFO: Using GPU-accelerated version")
    return predictions_desi_gpu(pred_sightlines, savefile, 
                                batch_size=128, 
                                max_windows_per_batch=4096)