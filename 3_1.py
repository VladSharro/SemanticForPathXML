import xml.etree.ElementTree as ET
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_similarity_metrics(similarity_matrix, ground_truth):
    # Convert similarity matrix to NumPy array
    similarity_matrix = np.array(similarity_matrix)

    # Convert ground_truth to NumPy array
    ground_truth = np.array(ground_truth)

    # Convert similarity matrix to binary by setting a threshold (e.g., 0.5)
    threshold = 0.3
    positive_matrix = (similarity_matrix >= threshold).astype(int)

    # Transpose the similarity matrix to match the shape of the ground truth
    positive_matrix = positive_matrix.T

    # Calculate True Positives (TP) and other metrics
    TP = np.sum(positive_matrix & ground_truth)
    FP = np.sum(positive_matrix & ~ground_truth)
    FN = np.sum(~positive_matrix & ground_truth)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score




def apply_selector(df):
    selector_df = pd.DataFrame(np.where(df >= 0.5, 1, 0), columns=df.columns, index=df.index)
    return selector_df




def print_element_tree(element, level=0, index=0):
    indent = " " * level
    print(f"{indent}{index} {element.tag}")
    for child in element:
        print_element_tree(child, level + 1, index + 1)


def jaccard_similarity(tags1, tags2):
    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = float(intersection) / union
    return similarity

def monge_elkan_similarity(element1, element2):
    scores = []
    for subelement1 in element1:
        subelement_scores = []
        for subelement2 in element2:
            subelement_score = fuzz.ratio(subelement1, subelement2) / 100.0
            subelement_scores.append(subelement_score)
        max_score = max(subelement_scores) if subelement_scores else 0.0
        scores.append(max_score)
    return sum(scores) / len(scores)









def parse_xml_paths1(xml_string):
    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()



def get_element_paths(element, path=''):
    paths = [f"{path}/{element.tag}"]
    for child in element:
        paths.extend(get_element_paths(child, f"{path}/{element.tag}"))
    return paths


def parse_xml_paths(element, path=''):
    paths = [f"{path}/{element.tag}"]
    for child in element:
        paths.extend(get_element_paths(child, f"{path}/{element.tag}"))
    return paths


mediated_scheme = [
    'Name',
    'Last Name',
    'First Name',
    'Team',
    'Team name',
    'Club',
    'Country',
    'Age',
    'Birthday',
    'B_Day',
    'Gender',
    'Salary',
    'Year',
    'Joined year'
    ]

nba = ['Name',
       'Rating',
       'Jersey',
       'Team',
       'Position',
       'B_Day',
       'Height',
       'Weight',
       'Salary',
       'Country',
       'BestPlayerOfTheYear',
       'College',
       'Version'
       ]




cycling = ['Year', 'Function', 'Last Name', 'First Name', 'Birth date', 'B_date_US', 'Age',
           'Gender', 'Category', 'Country', 'Continent', 'Team_Code', 'Team_Name', 'UCIID', 'Name'
           ]

fifa21 = ['ID', 'Name', 'Age', 'Country', 'Club', 'Position', 'Height', 'Weight', 'foot', 'Joined', 'Salary',
          'Contract Date', 'Finishing', 'Heading', 'Accuracy', 'Shootings', 'Dribbling', 'Sprint Speed', 'Shot Power',
          'Jumping', 'Penalties']


def create_xml_tree_med():
    root = ET.Element('Mediated')

    name_element = ET.SubElement(root, 'Name')
    for element in ['Name', 'Last_Name', 'First_Name', 'Age', 'Birthday', 'B_Day', 'Gender']:
        ET.SubElement(name_element, element)

    team_element = ET.SubElement(root, 'Team')
    for element in ['Team', 'Team_name', 'Club', 'Country', 'Salary', 'Year', 'Joined_year']:
        ET.SubElement(team_element, element)

    tree = ET.ElementTree(root)
    return tree

def create_xml_tree_nba():
    root = ET.Element('NBA')

    name_element = ET.SubElement(root, 'Name')
    for element in ['Name','B_Day', 'Height', 'Weight']:
        ET.SubElement(name_element, element)

    team_element = ET.SubElement(root, 'Team')
    for element in ['Rating','Jersey','Team','Position','Salary','Country','BestPlayerOfTheYear','College', 'Version']:
        ET.SubElement(team_element, element)
    tree = ET.ElementTree(root)
    return tree

def create_xml_tree_cycling():
    root = ET.Element('Cycling')

    name_element = ET.SubElement(root, 'Name')
    for element in ['Year', 'Function', 'Last_Name', 'First_Name', 'Birth_date', 'B_date_US', 'Age', 'Gender', 'Name']:
        ET.SubElement(name_element, element)

    team_element = ET.SubElement(root, 'Team')
    for element in ['Category', 'Country', 'Continent', 'Team_Code', 'Team_Name', 'UCIID']:
        ET.SubElement(team_element, element)
    tree = ET.ElementTree(root)
    return tree

def create_xml_tree_fifa21():
    root = ET.Element('Fifa21')

    name_element = ET.SubElement(root, 'Name')
    for element in ['Name', 'Age','Height', 'Weight']:
        ET.SubElement(name_element, element)

    team_element = ET.SubElement(root, 'Team')
    for element in ['ID', 'Country', 'Club', 'Position', 'foot', 'Joined', 'Salary',
          'Contract_Date', 'Finishing', 'Heading', 'Accuracy', 'Shootings', 'Dribbling', 'Sprint_Speed', 'Shot_Power',
          'Jumping', 'Penalties']:
        ET.SubElement(team_element, element)
    tree = ET.ElementTree(root)
    return tree





# Generate the XML tree
tree_med = create_xml_tree_med()
tree_nba = create_xml_tree_nba()
tree_cycling = create_xml_tree_cycling()
tree_fifa21 = create_xml_tree_fifa21()

root_med = tree_med.getroot()

#for element in root_med.iter():
    #print(element.tag)


print(tree_med)

# Write the XML tree to a file
tree_med.write('mediated_scheme_1.xml')
tree_nba.write('nba_scheme_1.xml')
tree_cycling.write('cycling_scheme_1.xml')
tree_fifa21.write('fifa21_scheme_1.xml')


mediated_tree = ET.parse('mediated_scheme_1.xml')
mediated_root = mediated_tree.getroot()
#print(mediated_root)

# Парсинг XML для 'nba'
nba_tree = ET.parse('nba_scheme_1.xml')
nba_root = nba_tree.getroot()

# Парсинг XML для 'cycling'
cycling_tree = ET.parse('cycling_scheme_1.xml')
cycling_root = cycling_tree.getroot()

# Парсинг XML для 'fifa21'
fifa21_tree = ET.parse('fifa21_scheme_1.xml')
fifa21_root = fifa21_tree.getroot()





print("Mediated")
print(print_element_tree(mediated_root))
print("NBA")

print(print_element_tree(nba_root))
print("Cycling")

print(print_element_tree(cycling_root))
print("FIFA 21")

print(print_element_tree(fifa21_root))

# Get the elements from the XML trees


med_path = get_element_paths(mediated_root)
nba_path = get_element_paths(nba_root)
cycling_path = get_element_paths(cycling_root)
fifa21_path = get_element_paths(fifa21_root)


for path in med_path:
    print(path)

for path in nba_path:
    print(path)

for path in cycling_path:
    print(path)

for path in fifa21_path:
    print(path)


jaccard_matrix1 = []
monge_elkan_matrix1 = []

jaccard_matrix2 = []
monge_elkan_matrix2 = []

jaccard_matrix3 = []
monge_elkan_matrix3 = []

print(len(med_path))
print(len(nba_path))

for med_paths in med_path:
    jaccard_similarities1 = []
    monge_elkan_similarities1 = []

    jaccard_similarities2 = []
    monge_elkan_similarities2 = []

    jaccard_similarities3 = []
    monge_elkan_similarities3 = []

    for nba_paths in nba_path:
        jaccard_similarities1.append(jaccard_similarity(med_paths, nba_paths))
        monge_elkan_similarities1.append(monge_elkan_similarity(med_paths, nba_paths))

    for cycling_paths in cycling_path:
        jaccard_similarities2.append(jaccard_similarity(med_paths, cycling_paths))
        monge_elkan_similarities2.append(monge_elkan_similarity(med_paths, cycling_paths))

    for fifa21_paths in fifa21_path:
        jaccard_similarities3.append(jaccard_similarity(med_paths, fifa21_paths))
        monge_elkan_similarities3.append(monge_elkan_similarity(med_paths, fifa21_paths))

    jaccard_matrix1.append(jaccard_similarities1)
    monge_elkan_matrix1.append(monge_elkan_similarities1)

    jaccard_matrix2.append(jaccard_similarities2)
    monge_elkan_matrix2.append(monge_elkan_similarities2)

    jaccard_matrix3.append(jaccard_similarities3)
    monge_elkan_matrix3.append(monge_elkan_similarities3)


print(jaccard_matrix1)
print(monge_elkan_matrix1)

print(jaccard_matrix2)
print(monge_elkan_matrix2)

print(jaccard_matrix3)
print(monge_elkan_matrix3)





data1 = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]

data2 = [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         ]

data3 = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]


print(len(med_path))
print(len(nba_path))
print(len(cycling_path))
print(len(fifa21_path))



# Your existing code...

# Convert the matrices to pandas DataFrames for visualization
jaccard_df1 = pd.DataFrame(jaccard_matrix1)
monge_elkan_df1 = pd.DataFrame(monge_elkan_matrix1)

jaccard_df2 = pd.DataFrame(jaccard_matrix2)
monge_elkan_df2 = pd.DataFrame(monge_elkan_matrix2)

jaccard_df3 = pd.DataFrame(jaccard_matrix3)
monge_elkan_df3 = pd.DataFrame(monge_elkan_matrix3)


precision1_1, recall1_1, f1_score1_1 = calculate_similarity_metrics(jaccard_matrix1, data1)
precision1_2, recall1_2, f1_score1_2 = calculate_similarity_metrics(monge_elkan_matrix1, data1)

precision2_1, recall2_1, f1_score2_1 = calculate_similarity_metrics(jaccard_matrix2, data2)
precision2_2, recall2_2, f1_score2_2 = calculate_similarity_metrics(monge_elkan_matrix2, data2)

precision3_1, recall3_1, f1_score3_1 = calculate_similarity_metrics(jaccard_matrix3, data3)
precision3_2, recall3_2, f1_score3_2 = calculate_similarity_metrics(monge_elkan_matrix3, data3)


print("Precision JaccardNBA:", precision1_1)
print("Recall JaccardNBA:", recall1_1)
print("F1 Score JaccardNBA:", f1_score1_1)

print("---------------------------------")

print("Precision MongeNBA:", precision1_2)
print("Recall MongeNBA:", recall1_2)
print("F1 Score MongeNBA:", f1_score1_2)

print("---------------------------------")

print("Precision JaccardCycling:", precision2_1)
print("Recall JaccardCycling:", recall2_1)
print("F1 Score JaccardCycling:", f1_score2_1)

print("---------------------------------")

print("Precision MongeCycling:", precision2_2)
print("Recall MongeCycling:", recall2_2)
print("F1 Score MongeCycling:", f1_score2_2)

print("---------------------------------")

print("Precision JaccardFIFA21:", precision3_1)
print("Recall JaccardFIFA21:", recall3_1)
print("F1 Score JaccardFIFA21:", f1_score3_1)

print("---------------------------------")

print("Precision MongeFIFA21:", precision3_2)
print("Recall MongeFIFA21:", recall3_2)
print("F1 Score MongeFIFA21:", f1_score3_2)





# Create heatmaps for Jaccard similarity matrices
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(jaccard_matrix1, cmap='hot', interpolation='nearest')
plt.title('Jaccard Similarity Mediated-NBA')
for i in range(len(med_path)):
    for j in range(len(nba_path)):
        plt.annotate(str(round(jaccard_matrix1[i][j], 2)), xy=(j, i), ha='center', va='center', color='black')

plt.subplot(1, 2, 2)
plt.imshow(jaccard_matrix2, cmap='hot', interpolation='nearest')
plt.title('Jaccard Similarity Mediated-Cycling')
for i in range(len(med_path)):
    for j in range(len(cycling_path)):
        plt.annotate(str(round(jaccard_matrix2[i][j], 2)), xy=(j, i), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(jaccard_matrix3, cmap='hot', interpolation='nearest')
plt.title('Jaccard Similarity Mediated-FIFA21')
for i in range(len(med_path)):
    for j in range(len(fifa21_path)):
        plt.annotate(str(round(jaccard_matrix3[i][j], 2)), xy=(j, i), ha='center', va='center', color='black')

plt.subplot(1, 2, 2)
plt.imshow(monge_elkan_matrix1, cmap='hot', interpolation='nearest')
plt.title('Monge-Elkan Similarity Mediated-NBA')
for i in range(len(med_path)):
    for j in range(len(nba_path)):
        plt.annotate(str(round(monge_elkan_matrix1[i][j], 2)), xy=(j, i), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(monge_elkan_matrix2, cmap='hot', interpolation='nearest')
plt.title('Monge-Elkan Similarity Mediated-Cycling')
for i in range(len(med_path)):
    for j in range(len(cycling_path)):
        plt.annotate(str(round(monge_elkan_matrix2[i][j], 2)), xy=(j, i), ha='center', va='center', color='black')

plt.subplot(1, 2, 2)
plt.imshow(monge_elkan_matrix3, cmap='hot', interpolation='nearest')
plt.title('Monge-Elkan Similarity Mediated-FIFA21')
for i in range(len(med_path)):
    for j in range(len(fifa21_path)):
        plt.annotate(str(round(monge_elkan_matrix3[i][j], 2)), xy=(j, i), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()