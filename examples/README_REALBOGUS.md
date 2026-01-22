# Real-Bogus Classification in STDPipe

This directory contains examples and tools for CNN-based real-bogus classification.

## Comprehensive Documentation

**For complete documentation, see:** [`../REALBOGUS.md`](../REALBOGUS.md)

The comprehensive guide includes:
- Overview and key concepts
- Stars-only default behavior
- Implementation details
- Command-line tool documentation
- Python API reference
- Training, testing, and evaluation guides
- Use cases and best practices
- Advanced features
- Complete workflow examples

## Files in This Directory

### Command-Line Tool

- **`train_realbogus.py`** - CLI tool for training, testing, and evaluating classifiers

**Quick start:**
```bash
# Train a classifier (stars-only by default)
python train_realbogus.py train --n-images 500 --epochs 30 --output model.h5

# Test on single image
python train_realbogus.py test --model model.h5 --image test.fits --interactive

# Evaluate performance
python train_realbogus.py evaluate --model model.h5 --n-images 100 --output results/
```

Run `python train_realbogus.py --help` for all options.

### Python Examples

- **`realbogus_example.py`** - Python API examples demonstrating various use cases

**Run examples:**
```bash
python realbogus_example.py
```

### Simulation Examples

- **`simulation_examples.py`** - Examples of simulating astronomical images

## Quick Links

- **Full Documentation**: [`../REALBOGUS.md`](../REALBOGUS.md)
- **API Reference**: See `help(realbogus.train_realbogus_classifier)` in Python
- **Online Docs**: https://stdpipe.readthedocs.io/

## Support

- **Issues**: https://github.com/karpov-sv/stdpipe/issues
- **Examples**: Run `python realbogus_example.py`
