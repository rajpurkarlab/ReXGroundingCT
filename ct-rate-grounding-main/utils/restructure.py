import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

t = {'Findings': {'Lungs/Airways/Pleura': ['The trachea and both main bronchi are unremarkable.', 'There is no evidence of obstructive pathology within the trachea and main bronchi.', 'Minimal bronchiectasis is present in the central regions of both lungs.', 'There are mild emphysematous changes in both lungs, with greater prominence in the upper lobes.', 'There is slight volume loss in the medial segment of the right middle lobe and the inferior subsegment of the lingular segment in the left upper lobe.', 'Multiple subcentimeter nonspecific nodules are scattered throughout both lungs.', 'No masses or infiltrative lesions are identified in either lung.'], 'Heart/Vessels': ['The heart contour and size are within normal limits, as are the calibers of the major mediastinal vascular structures.'], 'Mediastinum/Hila': ['Mediastinal evaluation is limited due to the lack of intravenous contrast.', 'There are small lymph nodes, each less than 1 cm in diameter, within the mediastinum and hilar regions.', 'No pathologically enlarged lymph nodes are noted.', 'The esophagus demonstrates no pathological wall thickening.'], 'Chest wall/Axilla': [], 'Lower neck': [], 'Bones': ['No lytic or destructive bone lesions are seen within the field of view.'], 'Upper abdomen': ['No masses or lesions are detectable in the upper abdominal organs within the limits of this non-contrast-enhanced CT scan.']}, 'Impressions': 'There are several subcentimeter nonspecific nodules in both lungs and minimal emphysematous changes, more pronounced in the upper lobes.'}

SHORTENED_NAME_MAPPING = {'Lungs/Airways/Pleura': 'L', 
               'Heart/Vessels': 'H', 
               'Mediastinum/Hila': 'M', 
               'Chest wall/Axilla': 'C',
               'Lower neck': 'N', 
               'Bones': 'B',
               'Upper abdomen': 'A'
            }

def restructure_for_extraction(translated_report):
    restructured_report = {'Findings': {}, 'Impressions': {}}
    
    for region, sentences in translated_report['Findings'].items():
        shortened_region_name = SHORTENED_NAME_MAPPING[region]
        for i, sentence in enumerate(sentences):
            restructured_report['Findings'][f'{shortened_region_name}{i}'] = sentence
    
    impressions_sentences = sent_tokenize(translated_report['Impressions'])
    restructured_report['Impressions'] = {f'I{j}': sentence for j, sentence in enumerate(impressions_sentences)}

    return str(restructured_report)

print(restructure_for_extraction(t))