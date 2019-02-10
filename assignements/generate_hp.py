

if __name__ == '__main__':
    lrs = [0.1, 0.01, 0.001]
    batch_sizes = [128, 256, 512]
    activations = ['linear', 'sigmoid', 'tanh']
    hidden_layers = [(256, 512), (512, 512), (512, 1024)]

    file = open('run_hp_search.sh','w')
    for lr in lrs:
        for bs in batch_sizes:
            for a in activations:
                for h1, h2 in hidden_layers:
                    file.write(f'python assignements/problem_1.py --batch_size {bs} --lr {lr} --activation {a} --h1 {h1} --h2 {h2}\n')
    file.close()
