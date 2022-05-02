
# # either a fully connected graph where it's the distance between the midpoint of the state
# # to the midpoint of the other state
# # or a graph where nodes share edges if they share a border, but that seems silly,
# # because alabama seems just as similar to kentucky as it is to florida
# # do it by zip code?

# # need to build a basic version of this, and yes it may result in some datapoints looking
# # like symmetric versions of each other
# # then move on to climate forward data
# # binning response storm amounts
# # then go lower level to zip code
# # # then connect it to housing prices

# can put this in my journal--> even if the predictive power is not there,
# i hope that my work can serve as a framework for someone else to use and potentially
# just swap in their own features into the design of the project
# (accounting for differences in temporal resolution, etc.)
#deleted water border between: new york and rhode island, 
# kept in michigan and minnesota because that water border seemed significant, climatically

def get_state_border_dict():
        
    my_newy = {
    "alabama": ["florida", "georgia", "mississippi", "tennessee"],
    'alaska': [], 
    'arizona': ['california', 'colorado', 'nevada', 'new mexico', 'utah'], 
    'arkansas': ['louisiana', 'mississippi', 'missouri', 'oklahoma', 'tennessee', 'texas'], 
    'california': ['arizona', 'nevada', 'oregon'], 
    'colorado': ['arizona', 'kansas', 'nebraska', 'new mexico', 'oklahoma', 'utah', 'wyoming'], 
    'connecticut': ['massachusetts', 'new york', 'rhode island'], 
    'delaware': ['maryland', 'new jersey', 'pennsylvania'], 
    'florida': ['alabama', 'georgia'], 
    'georgia': ['alabama', 'florida', 'north carolina', 'south carolina', 'tennessee'], 
    'hawaii': [], 
    'idaho': ['montana', 'nevada', 'oregon', 'utah', 'washington', 'wyoming'], 
    'illinois': ['indiana', 'iowa', 'michigan', 'kentucky', 'missouri', 'wisconsin'], 
    'indiana': ['illinois', 'kentucky', 'michigan', 'ohio'], 
    'iowa': ['illinois', 'minnesota', 'missouri', 'nebraska', 'south dakota', 'wisconsin'], 
    'kansas': ['colorado', 'missouri', 'nebraska', 'oklahoma'], 
    'kentucky': ['illinois', 'indiana', 'missouri', 'ohio', 'tennessee', 'virginia', 'west virginia'], 
    'louisiana': ['arkansas', 'mississippi', 'texas'], 
    'maine': ['new hampshire'], 
    'maryland': ['delaware', 'pennsylvania', 'virginia', 'west virginia'], 
    'massachusetts': ['connecticut', 'new hampshire', 'new york', 'rhode island', 'vermont'], 
    'michigan': ['illinois', 'indiana', 'minnesota', 'ohio', 'wisconsin'], 
    'minnesota': ['iowa', 'michigan', 'north dakota', 'south dakota', 'wisconsin'], 
    'mississippi': ['alabama', 'arkanssas', 'louisiana', 'tennessee'], 
    'missouri': ['arkansas', 'illinois', 'iowa', 'kansas', 'kentucky', 'nebraska', 'oklahoma', 'tennessee'], 
    'montana': ['idaho', 'north dakota', 'south dakota', 'wyoming'], 
    'nebraska': ['colorado', 'iowa', 'kansas', 'missouri', 'south dakota', 'wyoming'], 
    'nevada': ['arizona', 'california', 'idaho', 'oregon', 'utah'], 
    'new hampshire': ['maine', 'massachusetts', 'vermont'], 
    'new jersey': ['delaware', 'new york', 'pennsylvania'], 
    'new mexico': ['arizona', 'colorado', 'oklahoma', 'texas', 'utah'], 
    'new york': ['connecticut', 'massachusetts', 'new jersey', 'pennsylvania', 'vermont'], 
    'north carolina': ['north carolina ', 'georgia', 'south carolina', 'tennessee', 'virginia'], 
    'north dakota': ['minnesota', 'montana', 'south dakota'], 
    'ohio': ['indiana', 'kentucky', 'michigan', 'pennsylvania', 'west virginia'], 
    'oklahoma': ['arkansas', 'colorado', 'kansas', 'missouri', 'new mexico', 'texas'], 
    'oregon': ['california', 'idaho', 'nevada', 'washington'], 
    'pennsylvania': ['delaware', 'maryland', 'new jersey', 'new york', 'ohio', 'west virginia'], 
    'rhode island': ['connecticut', 'massachusetts', 'new york'], 
    'south carolina': ['georgia', 'north carolina'], 
    'south dakota': ['iowa', 'minnesota', 'montana', 'nebraska', 'north dakota', 'wyoming'], 
    'tennessee': ['alabama', 'arkansas', 'georgia', 'kentucky', 'mississippi', 'missouri', 'north carolina', 'virginia'], 
    'texas': ['arkansas', 'louisiana', 'new mexico', 'oklahoma'], 
    'utah': ['arizona', 'colorado', 'idaho', 'nevada', 'new mexico', 'wyoming'], 
    'vermont': ['massachusetts', 'new hampshire', 'new york'], 
    'virginia': ['kentucky', 'maryland', 'north carolina', 'tennessee', 'west virginia'], 
    'washington': ['idaho', 'oregon'], 
    'west virginia': ['west virginia', 'kentucky', 'maryland', 'ohio', 'pennsylvania', 'virginia'], 
    'wisconsin': ['illinois', 'iowa', 'michigan', 'minnesota'], 
    'wyoming': ['colorado', 'idaho', 'montana', 'nebraska', 'south dakota', 'utah']
        }
    return my_newy

         


# =============================================================================
# # import re
# re.sub(r'\b \b','no',mystr)
# re.sub(r'([a-zA-Z]+)(?=)', r'"\1"', mystr)
# mystr2 =  re.sub(r'([a-zA-Z]+)(?=)', r'"\1"', mystr)
# 
# =============================================================================
# =============================================================================
# for k,v in my_new_state_dict.items():
#     k = k.lower()
#     new_state_list = [state.lower() for state in v]
#     if k in new_state_list:
#         new_state_list.remove(k)
#     my_new_state_dict[k] = new_state_list
# 
# 
# =============================================================================

# state_to_neighbor_dict = {'arizona':['california', 'nevada', 'utah', 'new mexico'],
#                           'new mexico': ['arizona', 'texas', 'colorado'],
#                           'california': ['oregon', 'nevada', 'arizona'],
#                           'oregon': ['california', 'washington', 'idaho'],
#                           'washington': ['oregon', 'idaho'],
#                           'nevada': ['california', 'oregon', 'idaho', 'utah', 'arizona'],
#                           'idaho': ['utah', 'nevada', 'oregon', 'montana', 'wyoming'],
#                           'utah': ['arizona', 'california', 'nevada', 'idaho', 'wyoming', 'colorado'],
#                           'montana': ['wyoming', 'idaho', 'north dakota', 'south dakota'],
#                           'wyoming': ['colorado', 'utah', 'idaho', 'montana', 'north dakota', 'south dakota', 'nebraska' ]
#                           }

