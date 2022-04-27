import os
import glob
import sys
from pathlib import Path
import shutil
import re
import pandas as pd
import numpy as np
import shutil


def convert_dicom(folder, tmp_folder, regex_runs, regex_anat, throw_away_trs=5):
	# get the dcm2niix if not present
	if not Path("dcm2niix").exists():
		os.system("curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip")
		os.system("unzip dcm2niix_lnx.zip")
		os.system("rm dcm2niix_lnx.zip")

	print("convert_dicom", Path(folder).absolute())
	# remove old tmp folder
	if Path(tmp_folder).exists():
		os.system(f"rm -r {tmp_folder}")

	folders = []
	for dcm_filename in Path(folder).rglob("**/*.dcm"):
		print(dcm_filename)
		# we are searching for folders that contain dcm files, so if we already processed this folder, ignore it
		if dcm_filename.parent in folders:
			continue
		# remember the folder we already processed
		folder = dcm_filename.parent
		folders.append(folder)

		# ignore folders ending in _discard, as they contian the throw-away TRs
		if folder.name.endswith("_discard"):
			continue

		rename_pairs = []
		if re.match(regex_runs, folder.name):
			# print("func")

			target = Path(tmp_folder) / "func"

			# get the first 5 TRs to throw away
			number = int(folder.name.split("_")[-1])
			for i in range(throw_away_trs):
				file = folder / f"IM-{number:04d}-{i + 1:04d}.dcm"
				file2 = folder.parent / (folder.name + "_discard") / file.name
				file2.parent.mkdir(parents=True, exist_ok=True)
				# print(file, file.exists())
				if file.exists():
					rename_pairs.append([file, file2])

		elif re.match(regex_anat, folder.name):
			target = Path(tmp_folder) / "anat"
		else:
			target = Path(tmp_folder) / "misc"
			continue

		target.mkdir(parents=True, exist_ok=True)
		for from_file, to_file in rename_pairs:
			shutil.move(from_file, to_file)
		# sub-1_task-bird_run-1_bold
		os.system(f'./dcm2niix -f "%i_%p" -p y -z y -o "{target}" "{folder.absolute()}"')

def copy_nii_to_project(tmp_folder, project, outputfolder):
	# iterate over the folders
	for folder in Path(tmp_folder).iterdir():
		for file in folder.iterdir():
			target = None
			match = re.match(rf"{project}_(\d*)_Run(\d)_Birds\.(json|nii\.gz)", file.name)
			if match:
				sub, run, suffix = match.groups()
				target = f"{project}/sub-{sub}/func/sub-{sub}_task-bird_run-{run}_bold.{suffix}"
			match = re.match(rf"{project}_(\d*)_Run0_Localizer\.(json|nii\.gz)", file.name)
			if match:
				sub, suffix = match.groups()
				target = f"{project}/sub-{sub}/func/sub-{sub}_task-localizer_run-1_bold.{suffix}"
			match = re.match(rf"{project}_(\d*)_t1_mprage_sag_p2_iso1.0\.(json|nii\.gz)", file.name)
			if match:
				sub, suffix = match.groups()
				target = f"{project}/sub-{sub}/anat/sub-{sub}_T1w.{suffix}"
			target = Path(outputfolder) / target
			if target is None:
				print("ERROR", file)
			else:
				print("move", file, target)
				Path(target).parent.mkdir(parents=True, exist_ok=True)
				shutil.move(str(file), str(target))

	shutil.rmtree(tmp_folder)

def process_tsv_files(base_folder, project, regex_tsv, outputfolder, TR, throw_away_trs=5):
	# iterate over all tsv files
	for tsv_filename in Path(base_folder).rglob("**/*.tsv"):
		# try to extract sub task and run from the filename
		try:
			sub, task, run = re.match(regex_tsv, str(tsv_filename)).groups()
		except AttributeError:
			print("NO FIT", tsv_filename, file=sys.stderr)
			continue
		# read the tsv file and print
		df = pd.read_csv(tsv_filename, sep="\t")
		print(tsv_filename, sub, task, run, df.columns)
		# ensure that the onset column is at the beginning
		my_column = df.pop('onset')
		df.insert(0, "onset", my_column)
		# replace line breaks from the stimulus column
		df["stimulus"] = [s.replace("\n", "\\n").replace("\r", "").replace("\\", "/") for s in df["stimulus"]]
		# shift the time to account for the 5 throw-away TRs
		df["onset"] -= TR * throw_away_trs
		df["tr"] -= throw_away_trs
		df = df[~np.isnan(df.duration)]
		df = df.query("onset+duration >= 0")
		# skip empty dataframes
		if len(df) == 0:
			continue
		i = df.iloc[0].name
		onset = df.at[i, "onset"]
		# cut a bit into the first part if it extended beyond the first 5 throw-away TRs
		if onset < 0:
			difference = -onset
			df.at[i, "onset"] = 0
			df.at[i, "tr"] = 0
			df.at[i, "duration"] = df.at[i, "duration"] - difference
		# make sure the task name is set right
		if run == "0" or task == "localizer":
			task = "localizer"
		else:
			task = "bird"

		target_name = Path(outputfolder) / f"{project}/sub-{sub}/func/sub-{sub}_task-{task}_run-{run}_events.tsv"

		# if the folder has not been created yet it might have a wrong subject
		if Path(target_name).parent.exists():
			print(df.columns, target_name)
			df.to_csv(target_name, sep="\t", na_rep="n/a", index=False)