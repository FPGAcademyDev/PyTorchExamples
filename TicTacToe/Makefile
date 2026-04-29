.PHONY: train-sweep play-sweep summarize-sweep full-sweep

SWEEP_DIR := ai_sweeps/m03
EPOCHS := 50 100 150 200 250 300 350 400
HIDDENS := 32 64 128 256

train-sweep:
	mkdir -p $(SWEEP_DIR)
	for E in $(EPOCHS); do \
		for H in $(HIDDENS); do \
			python train.py --epochs $$E --hidden $$H --out $(SWEEP_DIR)/policy.e$$E.h$$H.pt > $(SWEEP_DIR)/train.e$$E.h$$H.txt; \
		done; \
	done
	
play-sweep:
	mkdir -p $(SWEEP_DIR)
	for E in $(EPOCHS); do \
		for H in $(HIDDENS); do \
			python play.py --weights $(SWEEP_DIR)/policy.e$$E.h$$H.pt --test-ai > $(SWEEP_DIR)/play.e$$E.h$$H.txt; \
		done; \
	done

summarize-sweep:
	python summarize_sweeps.py --input-dir $(SWEEP_DIR) --out $(SWEEP_DIR)/sweep_results.csv

full-sweep: train-sweep play-sweep summarize-sweep
