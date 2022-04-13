# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:18:51 2022

@author: 16028
"""


def get_fips_to_state_dict():
        
    fips_to_state_dict = {
    "01":"ALABAMA",
    "02":"ALASKA",
    "04":"ARIZONA",
    "05":"ARKANSAS",
    "06":"CALIFORNIA",
    "08":"COLORADO",
    "09":"CONNECTICUT",
    "10":"DELAWARE",
    "11":"DISTRICT OF COLUMBIA",
    "12":"FLORIDA",
    "13":"GEORGIA",
    "15":"HAWAII",
    "16":"IDAHO",
    "17":"ILLINOIS",
    "18":"INDIANA",
    "19":"IOWA",
    "20":"KANSAS",
    "21":"KENTUCKY",
    "22":"LOUISIANA",
    "23":"MAINE",
    "24":"MARYLAND",
    "25":"MASSACHUSETTS",
    "26":"MICHIGAN",
    "27":"MINNESOTA",
    "28":"MISSISSIPPI",
    "29":"MISSOURI",
    "30":"MONTANA",
    "31":"NEBRASKA",
    "32":"NEVADA",
    "33":"NEW HAMPSHIRE",
    "34":"NEW JERSEY",
    "35":"NEW MEXICO",
    "36":"NEW YORK",
    "37":"NORTH CAROLINA",
    "38":"NORTH DAKOTA",
    "39":"OHIO",
    "40":"OKLAHOMA",
    "41":"OREGON",
    "42":"PENNSYLVANIA",
    "44":"RHODE ISLAND",
    "45":"SOUTH CAROLINA",
    "46":"SOUTH DAKOTA",
    "47":"TENNESSEE",
    "48":"TEXAS",
    "49":"UTAH",
    "50":"VERMONT",
    "51":"VIRGINIA",
    "53":"WASHINGTON",
    "54":"WEST VIRGINIA",
    "55":"WISCONSIN",
    "56":"WYOMING",
    }
    
    return fips_to_state_dict


# state_to_fips_code_dict = {v: k for k, v in fips_to_state_dict.items()}
def get_state_to_fips_dict():
    
    state_to_fips_code_dict = {
    'ALABAMA': '01',
    'ALASKA': '02',
    'ARIZONA': '04',
    'ARKANSAS': '05',
    'CALIFORNIA': '06',
    'COLORADO': '08',
    'CONNECTICUT': '09',
    'DELAWARE': '10',
    'DISTRICT OF COLUMBIA': '11',
    'FLORIDA': '12',
    'GEORGIA': '13',
    'HAWAII': '15',
    'IDAHO': '16',
    'ILLINOIS': '17',
    'INDIANA': '18',
    'IOWA': '19',
    'KANSAS': '20',
    'KENTUCKY': '21',
    'LOUISIANA': '22',
    'MAINE': '23',
    'MARYLAND': '24',
    'MASSACHUSETTS': '25',
    'MICHIGAN': '26',
    'MINNESOTA': '27',
    'MISSISSIPPI': '28',
    'MISSOURI': '29',
    'MONTANA': '30',
    'NEBRASKA': '31',
    'NEVADA': '32',
    'NEW HAMPSHIRE': '33',
    'NEW JERSEY': '34',
    'NEW MEXICO': '35',
    'NEW YORK': '36',
    'NORTH CAROLINA': '37',
    'NORTH DAKOTA': '38',
    'OHIO': '39',
    'OKLAHOMA': '40',
    'OREGON': '41',
    'PENNSYLVANIA': '42',
    'RHODE ISLAND': '44',
    'SOUTH CAROLINA': '45',
    'SOUTH DAKOTA': '46',
    'TENNESSEE': '47',
    'TEXAS': '48',
    'UTAH': '49',
    'VERMONT': '50',
    'VIRGINIA': '51',
    'WASHINGTON': '53',
    'WEST VIRGINIA': '54',
    'WISCONSIN': '55',
    'WYOMING': '56'}
    return state_to_fips_code_dict