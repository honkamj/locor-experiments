"""Run evaluation."""

from application.execute import run_evaluation
from application.method import (
    ANTsApplication,
    CorrfieldApplication,
    LocorAblationStudyApplication,
    LocorApplication,
    NiftyRegMINDApplication,
    NiftyRegNMIApplication,
    SRWCRApplication,
)
from data.cermep import CERMEPDatasetInitializer
from data.ct_mr_thorax_abdomen import CTMRThoraxAbdomenDatasetInitializer
from data.ixi import IXIDatasetInitializer

run_evaluation(
    available_applications=[
        LocorApplication(),
        LocorAblationStudyApplication(),
        NiftyRegMINDApplication(),
        NiftyRegNMIApplication(),
        ANTsApplication(),
        CorrfieldApplication(),
        SRWCRApplication(),
    ],
    available_datasets=[
        CERMEPDatasetInitializer(),
        IXIDatasetInitializer(),
        CTMRThoraxAbdomenDatasetInitializer(mask_type="foreground_mask"),
        CTMRThoraxAbdomenDatasetInitializer(mask_type="roi_mask"),
    ],
)
