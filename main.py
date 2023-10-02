import os

from landscape import EnergyLandscape, DimensionReduction
from normalizing_flow import NormalizingFlow
from utils import parse_args, set_seed_everywhere, load_model


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    # set model path
    model_name = f"D={args.D}-rho1={args.rho_1}-rho2={args.rho_2}-{args.seed}"
    prefix = args.prefix + '-'
    if args.load_model_path != "":
        prefix += f"Load-{args.load_model_path}-"
    model_path = os.path.join(args.model_save_dir, prefix + model_name)
    if args.nf:
        model = NormalizingFlow(model_path, args)
    else:
        layers = [int(x) for x in args.hidden_sizes.split(',')]
        layers.append(1)
        if args.dr:
            layers.insert(0, 2)
            model = DimensionReduction(layers, model_path, args)
        else:
            layers.insert(0, args.dimension)
            model = EnergyLandscape(layers, model_path, args)

    # load model
    load_model_dir = os.path.join(args.load_model_path, 'model.pkl')
    if args.load_model_path == "":
        print("Training from scratch.")
    else:
        if os.path.exists(load_model_dir):
            print("Model exists.")
            load_model(model.dnn, load_model_dir)
        else:
            raise "Model does NOT exist."

    # train
    model.train()


if __name__ == "__main__":
    main()
