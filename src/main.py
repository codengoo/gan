import os
import random
import sys
import torch
import json

from cli.args import parse

if __name__ == "__main__":
    args = parse(sys.argv[1:])
    print(json.dumps(vars(args), indent=4))

    # Set training hardware
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpuid[0])

    # Set random
    torch_gen = torch.Generator()

    if args.seed:
        seed = args.seed
        print("You are using manual seed = {}".format(seed))

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch_gen.manual_seed(seed)
    else:
        print("Your are not using manual seed")

    # Set up output folder
    folder_output = os.path.join(args.output, args.dataset, args.experiment_name)
    print(folder_output)
    os.makedirs(folder_output, exist_ok=True)
    with open(os.path.join(folder_output, "args.txt"), "w") as text_file:
        text_file.write(json.dumps(vars(args), indent=4))
