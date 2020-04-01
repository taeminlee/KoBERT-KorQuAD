from argparse import ArgumentParser

from transformers.commands.serving import ServeCommand

from transformers import BertForQuestionAnswering
from tokenization_kobert import KoBertTokenizer

import os

if __name__ == '__main__':
    parser = ArgumentParser('Transformers CLI tool', usage='transformers-cli <command> [<args>]')
    commands_parser = parser.add_subparsers(help='transformers-cli command helpers')

    # Register commands
    ServeCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    # load model and tokenizer
    if(os.path.isdir(args.model)):
        model = BertForQuestionAnswering.from_pretrained('models')
    if(os.path.isdir(args.tokenizer)):
        tokenizer = KoBertTokenizer.from_pretrained('models')

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()