.PHONY: train-sweep play-sweep

train-sweep:
	for N in 50 100 150 200 250 300 350 400; do \
		python train.py --epochs $$N --out policy.$$N.pt > train.$$N.txt; \
	done
	
play-sweep:
	for N in 50 100 150 200 250 300 350 400; do \
		python play.py --weights policy.$$N.pt --test-ai > play.$$N.txt; \
	done
	
	
