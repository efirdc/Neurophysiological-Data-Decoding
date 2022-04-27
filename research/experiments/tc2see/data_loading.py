from nilearn.datasets import fetch_spm_auditory
from nilearn import image
from nilearn import masking, plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bids
from bids import BIDSLayout
import nilearn as ni
from pathlib import Path


class Data:
    def __init__(
            self,
            dataset_path,
            subject,
            task,
            run,
            space="MNI152NLin2009cAsym",
    ):
        self.subject = subject
        self.task = task
        self.run = run
        self.space = space

        bids.config.set_option('extension_initial_dot', True)
        self.layout = bids.BIDSLayout(Path(dataset_path) / "TC2See_prdgm")
        self.layout2 = bids.BIDSLayout(Path(dataset_path) / "derivatives_TC2See_prdgm", False)

        self.tr = self.layout2.get(datatype="func", extension="nii.gz", space=self.space, subject=self.subject, task=self.task, run=self.run, suffix="bold")[0].entities["RepetitionTime"]

    def cache(self, name, *args, **kwargs):
        property_name = "cache_"+name
        property = getattr(self, property_name, None)
        if property:
            return property
        filenames = self.layout2.get(*args, **kwargs)
        if len(filenames) > 1:
            raise
        property = ni.image.load_img(filenames[0])
        print(name, filenames[0])
        setattr(self, property_name, property)
        return property

    def cache_filename(self, name, filename):
        property_name = "cache_" + name
        property = getattr(self, property_name, None)
        if property:
            return property
        print(name, filename)
        property = filename.get_image()
        setattr(self, property_name, property)
        return property

    @property
    def anat(self):
        if self.space == "T1w":
            return self.cache("anat", datatype="anat", extension="nii.gz", suffix="T1w", space=None, subject=self.subject, return_type="file")
        else:
            return self.cache("anat", datatype="anat", extension="nii.gz", suffix="T1w", space=self.space, subject=self.subject, return_type="file")

    @property
    def gm_probseg(self):
        if self.space == "T1w":
            return self.cache_filename("gm_probseg", [d for d in self.layout2.get(datatype="anat", extension="nii.gz", space=None, subject=self.subject, suffix="probseg") if d.filename.endswith("label-GM_probseg.nii.gz")][0])
        else:
            return self.cache_filename("gm_probseg", [d for d in self.layout2.get(datatype="anat", extension="nii.gz", space=self.space, subject=self.subject, suffix="probseg") if d.filename.endswith("label-GM_probseg.nii.gz")][0])

    @property
    def fmri_img(self):
        return self.cache("fmri_img", datatype="func", extension="nii.gz", space=self.space, subject=self.subject, task=self.task, run=self.run, suffix="bold", return_type="file")

    @property
    def mask(self):
        return self.cache("mask", datatype="func", extension="nii.gz", space=self.space, subject=self.subject, task=self.task, run=self.run, suffix="mask", return_type="file")

    @property
    def events(self):
        if getattr(self, "_events", None) is not None:
            return self._events

        def category(text):
            text = text.replace("\\", "/")
            if text.startswith("docs/localizer/"):
                text = text.split("/")
                return text[2] + "_" + text[3]
            if text == "+":
                return "rest"
            if "annulus" in text:
                return "annulus"
            if "halfcircle" in text:
                return "halfcircle"
            if "docs/cropped" in text:
                return "bird"
            return None

        events = pd.read_csv(self.layout.get(task=self.task, subject=self.subject, run=self.run, extension=".tsv")[0].path, sep="\t")
        events.stimulus = [s.replace("\\", "/") for s in events.stimulus]
        if "class_id" in events.columns:
            events["genus"] = [np.nan if np.isnan(id) else "warbler" if id > 160 else "sparrow" for id in
                               events.class_id]
        events["trial_type"] = [category(s) for s in events.stimulus]
        self._events = events
        return self._events

    @property
    def confounds(self):
        if getattr(self, "_confound_df", None) is not None:
            return self._confound_df

        confound_files = self.layout2.get(datatype="func", extension="tsv", subject=self.subject, task=self.task, run=self.run, return_type="file")[0]
        confound_df = pd.read_csv(confound_files, delimiter='\t')
        self._confound_df = confound_df
        return self._confound_df


def load_data(subject, task, run, space="MNI152NLin2009cAsym"):

    def category(text):
        text = text.replace("\\", "/")
        if text.startswith("docs/localizer/"):
            text = text.split("/")
            return text[2] + "_" + text[3]
        if text == "+":
            return "rest"
        if "annulus" in text:
            return "annulus"
        if "halfcircle" in text:
            return "halfcircle"
        if "docs/cropped" in text:
            return "bird"
        return None

    bids.config.set_option('extension_initial_dot', True)
    layout = bids.BIDSLayout("project")
    layout2 = bids.BIDSLayout("derivatives3", False)
    #space = "MNI152NLin2009cAsym"
    #space = "T1w"

    tr = layout2.get(datatype="func", extension="nii.gz", space=space, subject=subject, task=task, run=run, suffix="bold")[0].entities["RepetitionTime"]

    fmri_img = ni.image.load_img(layout2.get(datatype="func", extension="nii.gz", space=space, subject=subject, task=task, run=run, suffix="bold", return_type="file")[0])
    mask = ni.image.load_img(layout2.get(datatype="func", extension="nii.gz", space=space, subject=subject, task=task, run=run, suffix="mask", return_type="file")[0])

    if space == "T1w":
        anat = ni.image.load_img(layout2.get(datatype="anat", extension="nii.gz", suffix="T1w", space=None, subject=subject, return_type="file")[0])
    else:
        anat = ni.image.load_img(layout2.get(datatype="anat", extension="nii.gz", suffix="T1w", space=space, subject=subject, return_type="file")[0])

    events = pd.read_csv(layout.get(task=task, subject=subject, extension=".tsv")[0].path, sep="\t")
    events.stimulus = [s.replace("\\", "/") for s in events.stimulus]
    if "class_id" in events.columns:
        events["genus"] = [np.nan if np.isnan(id) else "warbler" if id > 160 else "sparrow" for id in events.class_id]
    events["trial_type"] = [category(s) for s in events.stimulus]

    confound_files = layout2.get(datatype="func", extension="tsv", subject=subject, task=task, run=run, return_type="file")[0]
    confound_df = pd.read_csv(confound_files, delimiter='\t')
    #del localizer["tr"]
    #del localizer["stimulus"]
    return anat, fmri_img, events, mask, confound_df, tr


def tr_drop(fmri_img, events, confounds, tr):
    if 1:
        fmri_img = fmri_img.slicer[:, :, :, 1:]
        confounds = confounds.iloc[1:]
        events = events.iloc[1:].copy()
        events.onset -= tr
        events.tr -= 1
        return fmri_img, events, confounds
    fmri_img = fmri_img.slicer[:, :, :, 5:]
    confounds = confounds.iloc[5:]
    events = events.iloc[1:]
    events.onset -= events.iloc[0].onset
    events.tr -= 5
    return fmri_img, events, confounds

