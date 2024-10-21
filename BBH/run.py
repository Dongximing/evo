from utils import  set_seed
from args import parse_args
from llm_client import llm_init


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    client = None
    llm_config = llm_init("/Users/ximing/Desktop/EvoPrompt/BBH/auth.yaml", args.llm_type, args.setting)
    if args.evo_mode == 'de':
        from evoluter import DEEvoluter

        sampling_method = args.sampling_method
        evoluter = DEEvoluter(args, llm_config, client,sampling_method)
        evoluter.evolute()
    elif args.evo_mode == 'ga':
        from evoluter import GAEvoluter

        sampling_method = args.sampling_method
        evoluter = GAEvoluter(args, llm_config, client,sampling_method)
        evoluter.evolute()
    elif args.evo_mode == 'ape':
        from evoluter import ParaEvoluter
        evoluter = ParaEvoluter(args, llm_config, client)
        evoluter.evolute()
