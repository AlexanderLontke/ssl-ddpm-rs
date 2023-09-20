"""
This module contains baseline results
"""

# Scheibenreif et al.
SCHEIBENREIF_ET_AL_MULTILABEL = {
    ("Scheibenreif et al.", "mean"): {
        "val/multilabelf1score_Forest": 79,
        "val/multilabelf1score_Shrubland": 40,
        "val/multilabelf1score_Grassland": 47,
        "val/multilabelf1score_Wetlands": 24,
        "val/multilabelf1score_Croplands": 70,
        "val/multilabelf1score_Urban_Built-up": 80,
        "val/multilabelf1score_Barren": 34,
        "val/multilabelf1score_Water": 97,
    },
    ("Scheibenreif et al.", "std"): {
        "val/multilabelf1score_Forest": 2,
        "val/multilabelf1score_Shrubland": 3,
        "val/multilabelf1score_Grassland": 6,
        "val/multilabelf1score_Wetlands": 2,
        "val/multilabelf1score_Croplands": 3,
        "val/multilabelf1score_Urban_Built-up": 2,
        "val/multilabelf1score_Barren": 3,
        "val/multilabelf1score_Water": 0,
    },
}

SCHEIBENREIF_ET_AL_SEGMENTATION = {
    ("Scheibenreif et al.", "mean"): {
        "val/jaccardindexadapter_Forest": 62,
        "val/jaccardindexadapter_Shrubland": 48,
        "val/jaccardindexadapter_Grassland": 6,
        "val/jaccardindexadapter_Wetlands": 16,
        "val/jaccardindexadapter_Croplands": 52,
        "val/jaccardindexadapter_Urban_Built-up": 82,
        "val/jaccardindexadapter_Barren": 36,
        "val/jaccardindexadapter_Water": 99,
    },
    ("Scheibenreif et al.", "std"): {
        "val/jaccardindexadapter_Forest": 2,
        "val/jaccardindexadapter_Shrubland": 1,
        "val/jaccardindexadapter_Grassland": 0,
        "val/jaccardindexadapter_Wetlands": 1,
        "val/jaccardindexadapter_Croplands": 2,
        "val/jaccardindexadapter_Urban_Built-up": 0,
        "val/jaccardindexadapter_Barren": 1,
        "val/jaccardindexadapter_Water": 0,
    },
}
