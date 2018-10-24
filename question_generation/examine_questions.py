import json
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--question_file",
                    default="/home/celine/projects/clevr-dataset-gen/output/CLEVR_action_questions.json")

if __name__ == "__main__":
    args = parser.parse_args()
    file = json.load(open(args.question_file))
    for q in file['questions']:
        print('%s: %s %s' % (q['image'], str(q['answer']).ljust(15), q['question']))
