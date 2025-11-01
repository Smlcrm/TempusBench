#!/usr/bin/env python3
"""
Patch TensorBoard hparams plugin to fix make_tensor_proto compatibility issue.

This script patches TensorBoard's summary.py to use tf.compat.v1.make_tensor_proto
instead of tf.make_tensor_proto, which was removed in TensorFlow 2.x.

Run this script after installing TensorFlow and TensorBoard:
    python patch_tensorboard.py
"""
import os
import sys


def patch_tensorboard():
    """Patch TensorBoard hparams summary.py file."""
    try:
        import tensorboard
        tb_path = os.path.dirname(tensorboard.__file__)
        summary_path = os.path.join(tb_path, 'plugins', 'hparams', 'summary.py')
        
        if not os.path.exists(summary_path):
            print(f"✗ File not found: {summary_path}")
            return False
        
        # Read the file
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'tf.compat.v1.make_tensor_proto' in content:
            print(f"✓ Already patched: {summary_path}")
            return True
        
        # Check if needs patching
        if 'tf.make_tensor_proto' not in content:
            print(f"⚠ No make_tensor_proto found in file - may not need patching")
            return True
        
        # Create backup
        backup_path = summary_path + '.bak'
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✓ Created backup: {backup_path}")
        
        # Apply patch
        content = content.replace('tf.make_tensor_proto', 'tf.compat.v1.make_tensor_proto')
        
        # Write patched file
        with open(summary_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Successfully patched: {summary_path}")
        print("   Replaced tf.make_tensor_proto with tf.compat.v1.make_tensor_proto")
        return True
        
    except ImportError:
        print("✗ TensorBoard not installed. Install TensorBoard first:")
        print("   pip install tensorboard")
        return False
    except Exception as e:
        print(f"✗ Error patching TensorBoard: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = patch_tensorboard()
    sys.exit(0 if success else 1)


