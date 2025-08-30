import json

radiology_findings = {
    "S1": ["Unremarkable trachea and both main bronchi", "L0,L1", "N", "N"],
    "S2": ["Minimal bronchiectasis in the central regions of both lungs", "L2", "Y", "N"],
    "S3": ["Mild emphysematous changes in both lungs, more pronounced in the upper lobes", "L3,I0", "Y", "N"],
    "S4": ["Slight volume loss in the medial segment of the right middle lobe and the inferior subsegment of the lingular segment in the left upper lobe", "L4", "Y", "N"],
    "S5": ["Multiple subcentimeter nonspecific nodules scattered throughout both lungs", "L5,I0", "Y", "N"],
    "S6": ["No masses or infiltrative lesions in either lung", "L6", "N", "N"],
    "S7": ["Normal heart contour and size; normal calibers of major mediastinal vascular structures", "H0", "N", "N"],
    "S8": ["Limited mediastinal evaluation due to lack of intravenous contrast", "M0", "N", "N"],
    "S9": ["Small lymph nodes, each less than 1 cm in diameter, within the mediastinum and hilar regions", "M1", "N", "N"],
    "S10": ["No pathologically enlarged lymph nodes", "M2", "N", "N"],
    "S11": ["No pathological wall thickening in the esophagus", "M3", "N", "N"],
    "S12": ["No lytic or destructive bone lesions within the field of view", "B0", "N", "N"],
    "S13": ["No masses or lesions detectable in the upper abdominal organs on non-contrast-enhanced CT scan", "A0", "N", "N"]
}

with open('valid_970_a.json', 'w') as f:
    json.dump(radiology_findings, f)