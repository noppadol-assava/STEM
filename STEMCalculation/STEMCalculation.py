from featurev1 import *
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm
import os
import sys

NUMTHREAD= int(sys.argv[1])
approach = sys.argv[2]

with open(filepathhelper.path(dataset,'RandomForest.model'),'rb') as f:
    rfmodel = pickle.load(f)
rfmodel.set_params(n_jobs=None)
with open(filepathhelper.path(dataset,'normalize.dict'),'rb') as f:
    normalizedfeature = pickle.load(f)
def calculatefeature(teamissuecomponents):
    team=teamissuecomponents[0]
    issuekey=teamissuecomponents[1]
    component = teamissuecomponents[2]
    features = {}
    features.update(haibinfeaturecalculation(team))
    features['skilldiversity']=SkillDiversity(team,component,isTrain=False)
    features['skillcompetency']=SkillCompetency(team,component,isTrain=False)
    tis = team_interaction_score(team)
    features['team_positiveinteration']= tis['teamscorepos']
    features['team_negativeinteration'] = tis['teamscoreneg']
    
    atis = assignee_team_interaction_score(team)
    features['assignee_team_positiveinteration']=atis['teamscorepos']
    features['assignee_team_negativeinteration']=atis['teamscoreneg']
    
    tais = assignee_team_interaction_score(team)
    features['team_assignee_positiveinteration']=tais['teamscorepos']
    features['team_assignee_negativeinteration']=tais['teamscoreneg']
    
    features['numwork'] = NumWorkTogether(team)
    features['teamrelateness'] = TeamRelateness(team,1000,precal = True)
    features['assignee_teamrelatedness'] = AssigneeTeamRelateness(team,1000,precal = True)
    features['issuecloseness'] = IssueCloseness(team,1000,precal = True,isTrain=False)
    features['componentexperience'] = ComponentExperience(team,component,isTrain=False)
    features['issuefamiliarity'] = IssueFamiliarity(team,issuekey,fortest=True) 
    features['projectexperience'] = ProjectExperience(team,proj = re.match(r"(.*?)-", issuekey).group(1),isTrain=False)
    features['groupcontribution'] = getGroupContribution(team)
    features['CCR'] = CCR(team)
    features['CCSteiner'] = CCSteiner(team)
    features['CCSD'] = CCSD(team)
    features['CCLD'] = CCLD(team)
    #normalize
    for f in normalizedfeature:
        features[f] = (features[f] - normalizedfeature[f]['min'])/(normalizedfeature[f]['max']-normalizedfeature[f]['min'])
    inp_features = pd.DataFrame([features])[rfmodel.feature_names]
    prob = rfmodel.predict_proba(inp_features)[0][1]
    cost = 1/prob if prob>0 else math.inf
    features['cost']=cost
    return features

filesuffixname = '_'
if approach == 'Haibin':
    inpdir = '../../Haibin_approach'
    filesuffixname = filesuffixname + 'nonneghaibin_'
elif approach =='Random':
    inpdir = '../../random_approach'
    filesuffixname = filesuffixname + 'random_'
elif approach =='Dump':
    inpdir = '../../dump_approach'
    filesuffixname = filesuffixname + 'dump_'
elif approach =='Recent':
    inpdir = '../../recentness'
    filesuffixname = filesuffixname + 'recentness_'


if dataset.endswith('_hitnohit'):
    filesuffixname = filesuffixname + 'hitnohit_'
filesuffixname = filesuffixname + dataset.replace('_hitnohit','').lower()

with open(os.path.join(inpdir,'output'+filesuffixname+'.json')) as json_file:
    data = json.load(json_file)
# with open('output_our_hitnohit_moodle.json') as json_file:
#     data = json.load(json_file)

df_feature = {'issuekey':[],'rank':[]}
# featurelist = ['CCR','CCSteiner','CCSD','CCLD']
featurelist=['cost','experience','winexperience','winrate','roleexperience','closeness','connection','skilldiversity','skillcompetency','team_positiveinteration','team_negativeinteration','assignee_team_positiveinteration','assignee_team_negativeinteration','team_assignee_positiveinteration','team_assignee_negativeinteration','numwork','teamrelateness','assignee_teamrelatedness','issuecloseness','componentexperience','issuefamiliarity','projectexperience','groupcontribution','CCR','CCSteiner','CCSD','CCLD']
for i in featurelist:
    df_feature[i]=[]
teamissuecomponents = []
for res in tqdm(data):
    issuekey = res['issue']
    components = ItoC[issuekey] if issuekey in ItoC else []
    recs = res['r']
    for r in recs:
        df_feature['issuekey'].append(issuekey)
        df_feature['rank'].append(r['rank'])
        teamissuecomponents.append((r['team'],issuekey,components))


with mp.Pool(NUMTHREAD) as p:
    multi_out = tqdm(p.imap(calculatefeature,teamissuecomponents,chunksize=1),total=len(teamissuecomponents))
    results = [i for i in multi_out]
for result in results:
    for feature in result:
        df_feature[feature].append(result[feature])
df_feature = pd.DataFrame(df_feature)
df_feature.to_csv(os.path.join(inpdir,'recteam_feature'+filesuffixname+'.csv'))
