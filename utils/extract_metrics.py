import glob
import os.path
import statistics

root = ""
gans = ["ACGAN", "ADCGAN", "BigGAN", "ContraGAN", "MHGAN", "ReACGAN", "cStyleGAN2"]

for gan in gans:
    _is = []
    _fid = []
    _precision = []
    _recall = []
    _density = []
    _coverage = []
    _ifid = [[], [], [], []]
    for split in os.listdir(os.path.join(root, gan)):
        save_dir = os.path.join(root, gan, split)

        # Get newest evaluation
        eval_path = sorted(glob.glob(os.path.join(save_dir, "*_eval-*.out")))[-1]
        print(eval_path)

        eval_file = open(eval_path, "r")

        for line in eval_file.readlines():
            if "Inception score" in line:
                _is.append(float(line.split(" ")[-1]))
            elif "> FID score" in line:
                _fid.append(float(line.split(" ")[-1]))
            elif "Improved Precision" in line:
                _precision.append(float(line.split(" ")[-1]))
            elif "Improved Recall" in line:
                _recall.append(float(line.split(" ")[-1]))
            elif "Density" in line:
                _density.append(float(line.split(" ")[-1]))
            elif "Coverage" in line:
                _coverage.append(float(line.split(" ")[-1]))
            elif "> iFID score" in line:
                for i, fid in enumerate(line.split(" ")[-4:]):
                    fid = fid.replace(",", "")
                    fid = fid.replace("[", "")
                    fid = fid.replace("]", "")
                    _ifid[i].append(float(fid))

    print(
        f"{gan} & {statistics.mean(_is):.2f} & {statistics.mean(_fid):.2f} & {statistics.mean(_precision):.2f} & {statistics.mean(_recall):.2f} & {statistics.mean(_density):.2f} & {statistics.mean(_coverage):.2f}")
    print(
        f"{gan} & {statistics.mean(_ifid[0]):.2f} & {statistics.mean(_ifid[1]):.2f} & {statistics.mean(_ifid[2]):.2f} & {statistics.mean(_ifid[3]):.2f}")
