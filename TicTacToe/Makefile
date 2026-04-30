.PHONY: train-sweep play-sweep summarize-sweep full-sweep

# Policy architecture (1–7); override e.g. make train-sweep MODEL=7
MODEL := 1

SWEEP_DIR := ai_sweeps/m0$(MODEL)
EPOCHS := 400
HIDDENS := 32 64 128 256
# Checkpoints at 50, 100, ... up to E (matches train.py --out policy.e$E.h$H.pt + .e$E.h$H.pt)
CHECKPOINT_EPOCHS := $(shell seq 50 50 $(EPOCHS))

train-sweep:
	mkdir -p $(SWEEP_DIR)
	for H in $(HIDDENS); do \
		python train.py --epochs $(EPOCHS) --hidden $$H --model $(MODEL) --out $(SWEEP_DIR)/policy.pt > $(SWEEP_DIR)/train.h$$H.txt; \
	done

play-sweep:
	mkdir -p $(SWEEP_DIR)
	for E in $(CHECKPOINT_EPOCHS); do \
		for H in $(HIDDENS); do \
			python play.py --weights $(SWEEP_DIR)/policy.h$$H.e$$E.pt --model $(MODEL) --test-ai > $(SWEEP_DIR)/play.h$$H.e$$E.txt; \
		done; \
	done

summarize-sweep:
	python summarize_sweeps.py --input-dir $(SWEEP_DIR) --out $(SWEEP_DIR)/sweep_results.csv

full-sweep: train-sweep play-sweep summarize-sweep
