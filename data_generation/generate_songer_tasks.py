import os
import json
from tqdm import tqdm

from utils import get_cases_with_maj_opinion, save_opinions, subsample_and_save_decisions

states_file = 'songer_codes/songer_states.txt'

header = "What follows is an opinion from a United States Court of Appeals."

states_instructions = 'Answer with the name of the state, or one of the following territories: ' \
    'District of Columbia, Puerto Rico, Virgin Islands, Panama Canal Zone, or "not applicable" or "not determined".'
state_fill = ''

tasks_general = {
    'method': {
        'name': 'songer_method',
        'instruction': f'{header} Your task is to determine the nature of the proceeding in the court of appeals for the case, that is, ' \
            'the legal history of the case, indicating whether there had been prior appellate court proceeding on the same ' \
            'case prior to the decision currently coded. ' \
            'Assume that the case had been decided by the panel for the first time if there was no indication to the ' \
            'contrary in the opinion. ' \
            'The opinion usually, but not always, explicitly indicates when a decision was made "en banc" (though the spelling of ' \
            '"en banc" varies). However, if more than 3 judges were listed as participating in the decision, code the decision ' \
            'as enbanc even if there was no explicit description of the proceeding as en banc.',
        'question': 'What is the nature of the proceeding in the court of appeals for this case?',
        'answer_choices': {
            1: 'decided by panel for first time (no indication of re-hearing or remand)',
            2: 'decided by panel after re-hearing (second time this case has been heard by this same panel)',
            3: 'decided by panel after remand from Supreme Court',
            4: 'decided by court en banc, after single panel decision',
            5: 'decided by court en banc, after multiple panel decisions',
            6: 'decided by court en banc, no prior panel decisions',
            7: 'decided by panel after remand to lower court',
            8: 'other',
            9: 'not ascertained',
        }
    },
    'circuit': {
        'name': 'songer_circuit',
        'instruction': f'{header} Your task is to identify the circuit of the court that decided the case.',
        'question': 'What is the circuit of the court that decided the case?',
        'answer_choices': {
            1: 'First Circuit', 2: 'Second Circuit', 3: 'Third Circuit',
            4: 'Fourth Circuit', 5: 'Fifth Circuit', 6: 'Sixth Circuit',
            7: 'Seventh Circuit', 8: 'Eighth Circuit', 9: 'Ninth Circuit',
            10: 'Tenth Circuit', 11: 'Eleventh Circuit', 0: 'District of Columbia Circuit',

        }
    },
    'state': {
        'name': 'songer_state',
        'instruction': f'{header} Your task is to identify the state or territory in which the case was first heard. ' \
            'If the case began in the federal district court, consider the state of that district court. ' \
            'If it is a habeas corpus case, consider the state of the state court that first heard the case. ' \
            f'If the case originated in a federal administrative agency, answer "not applicable". {states_instructions}',
        'question': 'In what state or territory was the case first heard?',
        'answer_choices': states_file,
    },
    'district': {
        'name': 'songer_district',
        'instruction': f'{header} Your task is to identify which district in the state {state_fill} the case came from. ' \
            'If the case did not come from a federal district court, answer "not applicable".',
        'question': 'From which district in the state was this case appealed?',
        'answer_choices': {
            0: 'Not applicable',
            1: 'Eastern',
            2: 'Western',
            3: 'Central',
            4: 'Middle',
            5: 'Southern',
            6: 'Northern',
            7: 'Whole state is one judicial district',
            8: 'Not ascertained',
        }
    },
    'origin': {
        'name': 'songer_origin',
        'instruction': f'{header} Your task is to identify the type of court which made the original ' \
            'decision. Code cases removed from a state court as originating in federal district court. ' \
            'For "State court", include habeas corpus petitions after conviction in state court and petitions ' \
            'from courts of territories other than the U.S. District Courts. ' \
            'For "Special DC court", include courts other than the US District Court for DC. ' \
            'For "Other", include courts such as the Tax Court and a court martial.',
        'question': 'What type of court made the original decision?',
        'answer_choices': {
            1: 'Federal district court (single judge)',
            2: '3 judge district court',
            3: 'State court',
            4: 'Bankruptcy court, referee in bankruptcy, special master',
            5: 'Federal magistrate',
            6: 'Federal administrative agency',
            7: 'Special DC court',
            8: 'Other ',
            9: 'Not ascertained',
        }
    },
    'source': {
        'name': 'songer_source',
        'instruction': f'{header} Your task is to identify the forum that heard this case immediately before ' \
            'the case came to the court of appeals.',
        'question': 'What forum heard this case immediately before the case came to the court of appeals?',
        'answer_choices': {
            1: 'Federal district court (single judge)',
            2: '3 judge district court',
            3: 'State court',
            4: 'Bankruptcy court, referee in bankruptcy, special master',
            5: 'Federal magistrate',
            6: 'Federal administrative agency',
            7: 'Court of Customs & Patent Appeals',
            8: 'Court of Claims',
            9: 'Court of Military Appeals',
            10: 'Tax Court or Tax Board',
            11: 'Administrative law judge',
            12: 'U.S. Supreme Court (remand)',
            13: 'Special DC court (not the US District Court for DC)',
            14: 'Earlier appeals court panel',
            15: 'Other',
            16: 'Not ascertained',
        }
    },
    'applfrom': {
        'name': 'songer_applfrom',
        'instruction': f'{header} Your task is to identify the type of district court decision or ' \
            'judgment appealed from (i.e., the nature of the decision below in the district court).',
        'question': 'What is the type of district court decision or ' \
            'judgment appealed from (i.e., the nature of the decision below in the district court)?',
        'answer_choices': {
            1: 'Trial (either jury or bench trial)',
            2: 'Injunction or denial of injunction or stay of injunction',
            3: 'Summary judgment or denial of summary judgment',
            4: 'Guilty plea or denial of motion to withdraw plea',
            5: 'Dismissal (include dismissal of petition for habeas corpus)',
            6: 'Appeals of post judgment orders (e.g., attorneys\' fees, costs, damages, JNOV - judgment nothwithstanding the verdict)',
            7: 'Appeal of post settlement orders',
            8: 'Not a final judgment: interlocutory appeal',
            9: 'Not a final judgment: mandamus',
            10: 'Other (e.g., pre-trial orders, rulings on motions, directed verdicts) or could not determine nature of final judgment',
            11: 'Does not fit any of the above categories, but opinion mentions a "trial judge"',
            12: 'Not applicable (e.g., decision below was by a federal administrative agency, tax court)',
        },
    },
    'adminrev': {
        'name': 'songer_adminrev',
        'instruction': f'{header} Your task is to identify the federal agency (if any) whose decision ' \
            'was reviewed by the court of appeals. If there was no prior agency ' \
            'action, choose "not applicable".',
        'question': 'What federal agency\'s decision was reviewed by the court of appeals?',
        'answer_choices': {
            1: 'Benefits Review Board',
            2: 'Civil Aeronautics Board',
            3: 'Civil Service Commission',
            4: 'Federal Communications Commission',
            5: 'Federal Energy Regulatory Commission',
            6: 'Federal Power Commission',
            7: 'Federal Maritime Commission',
            8: 'Federal Trade Commission',
            9: 'Interstate Commerce Commission',
            10: 'National Labor Relations Board',
            11: 'Atomic Energy Commission',
            12: 'Nuclear Regulatory Commission',
            13: 'Securities & Exchange Commission',
            14: 'Other federal agency',
            15: 'Not ascertained or not applicable',
        }
    },
    'opinstat': {
        'name': 'songer_opinstat',
        'instruction': f'{header} Your task is to identify whether the opinion writter is identified in the ' \
            'opinion or whether the opinion was per curiam.',
        'question': 'Is the opinion writer identified in the opinion, or was the opinion per curiam?',
        'answer_choices': {
            1: 'Signed, with reasons',
            2: 'Per curiam, with reasons',
            9: 'Not ascertained',
        }
    },
    'classact': {
        'name': 'songer_classact',
        'instruction': f'{header} Your task is to determine whether the case is described in the opinion as a ' \
            'class action suit. If so, the opinion should specifically indicate that the action was filed ' \
            'as a representative of a class or of "all others similarly situated".',
        'question': 'Is the case described in the opinion as a class action suit?',
        'answer_choices': {
            0: 'No',
            1: 'Yes',
        }
    },
    'crossapp': {
        'name': 'songer_crossapp',
        'instruction': f'{header} Your task is to determine whether there were cross appeals ' \
            'from the decision below to the court of appeals that were consolidated in the present case.',
        'question': 'Were there cross appeals from the decision below to the court of appeals that ' \
            'were consolidated in the present case?',
        'answer_choices': {
            0: 'No',
            1: 'Yes',
            2: 'Not ascertained',
        }
    },
    # not including sanction since all except two are classified as 'not ascertained'
    'initiate': {
        'name': 'songer_initiate',
        'instruction': f'{header} Your task is to identify what party initiated the appeal. ' \
            'For cases with cross appeals or multiple docket numbers, if the opinion does not ' \
            'explicitly indicate which appeal was filed first, assumes that the first litigant listed as the ' \
            '"appellant" or "petitioner" was the first to file the appeal. ' \
            'In federal habeas corpus petitions, consider the prisoner to be the plaintiff.',
        'question': 'What party initiated the appeal?',
        'answer_choices': {
            1: 'Original plaintiff',
            2: 'Original defendant',
            3: 'Federal agency representing plaintiff',
            4: 'Federal agency representing defendant',
            5: 'Intervenor',
            8: 'Not applicable',
            9: 'Not ascertained',
        },
    }
}

header_participants = 'Intervenors who participated as parties at the courts of appeals should be ' \
    'counted as either appellants or respondents when it can be determined whose position they ' \
    'supported. For example, if there were two plaintiffs who lost in ' \
    'district court, appealed, and were joined by four intervenors who ' \
    'also asked the court of appeals to reverse the district court, the ' \
    'number of appellants should be coded as six.'

head_appellants = 'In some cases there is some confusion over who should be ' \
    'listed as the appellant and who as the respondent. This confusion ' \
    'is primarily the result of the presence of multiple docket numbers ' \
    'consolidated into a single appeal that is disposed of by a single ' \
    'opinion. Most frequently, this occurs when there are cross appeals ' \
    'and/or when one litigant sued (or was sued by) multiple litigants ' \
    'that were originally filed in district court as separate actions. ' \
    'The coding rule followed in such cases should be to go strictly by the ' \
    'designation provided in the title of the case. The first person ' \
    'listed in the title as the appellant should be coded as the appellant ' \
    'even if they subsequently appeared in a second docket number as the ' \
    'respondent and regardless of who was characterized as the appellant ' \
    'in the opinion.\n' \
    'To clarify the coding conventions, consider the following ' \
    'hypothetical case in which the US Justice Department sues a labor ' \
    'union to strike down a racially discriminatory seniority system and ' \
    'the corporation (siding with the position of its union) ' \
    'simultaneously sues the government to get an injunction to block ' \
    'enforcement of the relevant civil rights law. From a district ' \
    'court decision that consolidated the two suits and declared the ' \
    'seniority system illegal but refused to impose financial penalties ' \
    'on the union, the corporation appeals and the government and union ' \
    'file cross appeals from the decision in the suit brought by the ' \
    'government. Assume the case was listed in the Federal Reporter as ' \
    'follows:\n' \
        'United States of America,\n' \
        'Plaintiff, Appellant\n' \
        'v\n' \
        'International Brotherhood of Widget Workers,AFL-CIO\n' \
        'Defendant, Appellee.\n' \
        'International Brotherhood of Widget Workers,AFL-CIO\n' \
        'Defendants, Cross-appellants\n' \
        'v\n' \
        'United States of America.\n' \
        'Widgets, Inc. & Susan Kuersten Sheehan, President & Chairman\n' \
        'of the Board\n' \
        'Plaintiff, Appellants,\n' \
        'v\n' \
        'United States of America,\n' \
        'Defendant, Appellee.\n' \
    'This case should be coded as follows:' \
    'Appellant = United States, ' \
    'Respondents = International Brotherhood of Widget Workers Widgets, Inc., ' \
    'Total number of appellants = 1, ' \
    'Number of appellants that fall into the category "the federal government, its agencies, and officials" = 1, ' \
    'Total number of respondents = 3, ' \
    'Number of respondents that fall into the category "private business and its executives" = 2, ' \
    'Number of respondents that fall into the category "groups and associations" = 1.' \

header_specific_app = 'Note that if an individual is listed by name, but their ' \
    'appearance in the case is as a government official, then they should be ' \
    'counted as a government rather than as a private person. For ' \
    'example, in the case "Billy Jones & Alfredo Ruiz v Joe Smith" where ' \
    'Smith is a state prisoner who brought a civil rights suit against ' \
    'two of the wardens in the prison (Jones & Ruiz), the following ' \
    'values should be coded: number of appellants that fall into the ' \
    'category "natural persons" =0 and number that fall into the category ' \
    '"state governments, their agencies, and officials" =2. A similar logic ' \
    'should be applied to businesses and associations. Officers of a company ' \
    'or association whose role in the case is as a representative of ' \
    'their company or association should be coded as being a business or ' \
    'association rather than as a natural person. However, employees of ' \
    'a business or a government who are suing their employer should be coded ' \
    'as natural persons. Likewise, employees who are charged with ' \
    'criminal conduct for action that was contrary to the company ' \
    'policies should be considered natural persons.\n' \
    'If the title of a case listed a corporation by name and then ' \
    'listed the names of two individuals that the opinion indicated were ' \
    'top officers of the same corporation as the appellants, then the ' \
    'number of appellants should be coded as three and all three were coded as ' \
    'a business (with the identical detailed code). Similar logic should be ' \
    'applied when government officials or officers of an association ' \
    'were listed by name.'

header_nature_participants = 'When coding the detailed nature of participants, ' \
    'use your personal knowledge about the ' \
    'participants, if you are completely confident of the accuracy of ' \
    'your knowledge, even if the specific information is not in ' \
    'the opinion. For example, if "IBM" is listed as the appellant it ' \
    'could be classified as "clearly national or international in scope" ' \
    'even if the opinion did not indicate the scope of the business. ' \

# for the first appellant and the second appellant! APPEL1 and APPEL2, RESPOND1, RESPOND2
# this is constructed using GENAPP1, GENAPP2, GENRESP1, GENRESP2
party_details = {
    1: {  # general category 1
        2: {
            'instruction': 'Your task is to classify the scope of this business into one of the ' \
                'following categories: "local" (individual or family owned business, scope ' \
                    'limited to single community; generally proprietors, who are not incorporated); ' \
                '"neither local nor national" (e.g., an ' \
                    'electrical power company whose operations cover one-third of the state); ' \
                '"national or multi-national" (assume that insurance companies and ' \
                    'railroads are national in scope); and ' \
                '"not ascertained".',
            'question': 'What is the scope of this business?',
            'answer_choices': {
                1: 'local',
                2: 'neither local nor national',
                3: 'national or multi-national',
                4: 'not ascertained',
            },
        },
        3: {
            'instruction': 'Your task is to determine what category of business best describes ' \
                'the area of activity of this litigant which is involved in this case.',
            'question': 'What category of business best describes the area of activity of this ' \
                'litigant which is involved in this case?',
            'answer_choices': {
                1: 'agriculture',
                2: 'mining',
                3: 'construction',
                4: 'manufacturing',
                5: 'transportation',
                6: 'trade',
                7: 'financial institution',
                8: 'utilities',
                9: 'other',
                0: 'unclear',
            }
        },
        4: {
            'instruction': 'Your task is to determine what subcategory of business best describes this litigant.',
            'question': 'What subcategory of business best describes this litigant?',
            'choosing_rule': lambda x: str(x)[2],  # third digit
            'possible_choices': {
                1: {
                    1: 'single family farm',
                    2: 'commercial farm, agri-business',
                    3: 'farm - other ',
                    0: 'unclear',
                },
                2: {
                    1: 'oil and gas',
                    2: 'coal',
                    3: 'metals',
                    4: 'other ',
                    0: 'unclear',
                },
                3: {
                    1: 'residential',
                    2: 'commercial or industrial',
                    3: 'other',
                    0: 'unclear'
                },
                4: {
                    1: 'auto',
                    2: 'chemical',
                    3: 'drug',
                    4: 'food processing',
                    5: 'oil refining',
                    6: 'textile',
                    7: 'electronic',
                    8: 'alcohol or tobacco',
                    9: 'other',
                    0: 'unclear',
                },
                5: {
                    1: 'railroad',
                    2: 'boat, shipping',
                    3: 'shipping freight, UPS, flying tigers',
                    4: 'airline',
                    5: 'truck, armored cars',
                    6: 'other',
                    0: 'unclear',
                },
                6: {
                    1: 'auto, auto parts, auto repairs',
                    2: 'chemical',
                    3: 'drug',
                    4: 'food',
                    5: 'oil, natural gas, gasoline',
                    6: 'textile, clothing',
                    7: 'electronic',
                    8: 'alcohol or tobacco',
                    9: 'general merchandise',
                    10: 'other ',
                    0: 'unclear',
                },
                7: {
                    1: 'bank',
                    2: 'insurance',
                    3: 'savings and loan',
                    4: 'credit union',
                    6: 'other pension fund',
                    7: 'other financial institution or investment company',
                    0: 'unclear',
                },
                8: {
                    1: 'nuclear power plants',
                    2: 'other producers of power',
                    3: 'telephone',
                    4: 'other utilities',
                    0: 'unclear',
                },
                9: {
                    1: 'medical clinics, health organizations, nursing homes, ' \
                        'medical doctors, medical labs, or other private health ' \
                        'care facilities',
                    2: 'private attorney or law firm',
                    3: 'media - including magazines, newspapers, radio & TV ' \
                        'stations and networks, cable TV, news organizations',
                    4: 'school - for profit private educational enterprise ' \
                        '(including business and trade schools)',
                    5: 'housing, car, or durable goods rental or lease',
                    6: 'entertainment: amusement parks, race tracks, for profit ' \
                        'camps, record companies, movie theaters and producers, ' \
                        'ski resorts, hotels, restaurants, etc.',
                    7: 'information processing',
                    8: 'consulting',
                    9: 'security and/or maintenance service',
                    10: 'other service (including accounting)',
                    11: 'other (including a business pension fund)',
                    0: 'unclear',
                },
                0: {
                    1: 'auto industry',
                    2: 'chemical industry',
                    3: 'drug industry',
                    4: 'food industry',
                    5: 'oil & gas industry',
                    6: 'clothing & textile industry',
                    7: 'electronic industry',
                    8: 'alcohol and tobacco industry',
                    9: 'other',
                    0: 'unclear'
                },
            }
        }
    },
    2: {
        2: {
            'instruction': 'Your task is to determine what category of private associations best describes this litigant.',
            'question': 'What category of private associations best describes this litigant?',
            'answer_choices': {
                1: 'business, trade, professional, or union (BTPU)',
                2: 'other',
            },
        },
        #  describe specific subcategories of organizations
        3: {
            'instruction': 'Your task is to determine what subcategory of private association best describes this litigant.',
            'question': 'What subcategory of private association best describes this litigant?',
            'choosing_rule': lambda x: str(x)[1],  # second digit
            'possible_choices': {
                1: {
                    1: 'Business or trade association',
                    2: 'utilities co-ops',
                    3: 'Professional association - other than law or medicine',
                    4: 'Legal professional association',
                    5: 'Medical professional association',
                    6: 'AFL-CIO union (private)',
                    7: 'Other private union',
                    8: 'Private Union - unable to determine whether in AFL-CIO',
                    9: 'Public employee union- in AFL-CIO ' \
                        '(include groups called professional organizations if ' \
                        'their role includes bargaining over wages and work conditions)',
                    10: 'Public Employee Union - not in AFL-CIO',
                    11: 'Public Employee Union - unable to determine if in AFL-CIO',
                    12: 'Union pension fund; other union funds (e.g., vacation funds)',
                    13: 'Other',
                    0: 'Unclear',
                },
                2: {
                    1: 'Civic, social, fraternal organization',
                    2: 'Political organizations - Other than political parties ' \
                        'Examples: Civil rights focus; Public Interest - broad, ' \
                        'civil liberties focus (ACLU) or broad, multi-issue focus ' \
                        '(Common Cause, Heritage Foundation, ADA) or single issue ' \
                        '- Environmental ENV, Abortion, etc. (prolife, ' \
                        'pro-abortion), elderly, consumer interests: Consumer ' \
                        'Federation of America, Consumer\'s Union, National ' \
                        'Railroad Passenger Association; PAC',
                    3: 'Political party',
                    4: 'Educational organization - Private, non-profit school',
                    5: 'Educational organization - Association, not individual school - PTA or PTO',
                    6: 'Religious or non-profit hospital or medical care facility (e.g., nursing home)',
                    7: 'Other religious organization (includes religious foundations)',
                    8: 'Charitable or philanthropic organization (including ' \
                        'foundations, funds, private museums, private libraries)',
                    9: 'Other',
                    0: 'Unclear'
                }
            }
        }
    },
    3: {
        2: {
            'instruction': 'Your task is to determine which category of federal government agencies and activities best describes this litigant.',
            'question': 'Which category of federal government agencies and activities best describes this litigant?',
            'answer_choices': {
                1: 'cabinet level department',
                2: 'courts or legislative',
                3: 'agency whose first word is "federal"',
                4: 'other agency, beginning with "A" thru "E"',
                5: 'other agency, beginning with "F" thru "N"',
                6: 'other agency, beginning with "O" thru "R"',
                7: 'other agency, beginning with "S" thru "Z"',
                8: 'Distric of Columbia',
                9: 'other, not listed, not able to classify',
            },
        },
        3: {
            'instruction': 'Your task is to determine which specific federal government agency best describes this litigant.',
            'question': 'Which specific federal government agency best describes this litigant?',
            'choosing_rule': lambda x: str(x)[1],  # second digit
            'possible_choices': {
                1: {
                    1: 'Department of Agriculture',
                    2: 'Department of Commerce',
                    3: 'Department of Defense (includes War Department and Navy Department)',
                    4: 'Department of Education',
                    5: 'Department of Energy',
                    6: 'Department of Health, Education and Welfare',
                    7: 'Department of Health & Human Services',
                    8: 'Department of Housing and Urban Development',
                    9: 'Department of Interior',
                    10: 'Department of Justice (does not include FBI or parole boards; does include US Attorneys)',
                    11: 'Department of Labor (except OSHA)',
                    12: 'Post Office Department',
                    13: 'Department of State',
                    14: 'Department of Transportation, National Transportation Safety Board',
                    15: 'Department of the Treasury (except IRS)',
                    16: 'Department of Veterans Affairs',
                },
                2: {
                    1: 'one or both houses of Congress',
                    2: 'congressional committee',
                    3: 'officer of Congress or other Congress related actor',
                    4: 'Federal District Court (or judge)',
                    5: 'Federal Circuit Court of Appeals (or judge)',
                    6: 'Court of Claims (or judge)',
                    7: 'Tax Court (or judge)',
                    8: 'Bankruptcy Court (or judge)',
                    9: 'other court or judge',
                },
                3: {
                    1: 'Federal Aviation Administration',
                    2: 'Federal Bureau of Investigation (FBI)',
                    3: 'Federal Coal Mine Safety Board',
                    4: 'Federal Communications Commission',
                    5: 'Federal Deposit Insurance Corporation and FSLIC',
                    6: 'Federal Election Commission',
                    7: 'Federal Energy Agency (Federal Power Commission)',
                    8: 'Federal Energy Regulatory Commission',
                    9: 'Federal Home Loan Bank Board',
                    10: 'Federal Housing Authority (FHA)',
                    11: 'Federal Labor Relations Authority',
                    12: 'Federal Maritime Board',
                    13: 'Federal Maritime Commission',
                    14: 'Federal Mine Safety & Health Administration',
                    15: 'Federal Mine Safety & Health Review Commission',
                    16: 'Federal Reserve System',
                    17: 'Federal Trade Commission',
                },
                4: {
                    1: 'Benefits Review Board',
                    2: 'Civil Aeronautics Board',
                    3: 'Civil Service Commission (U.S.)',
                    4: 'Commodity Futures Trading Commission',
                    5: 'Consumer Products Safety Commission',
                    6: 'Copyright Royalty Tribunal',
                    7: 'Drug Enforcement Agency',
                    8: 'Environmental Protection Agency',
                    9: 'Equal Employment Opportunity Commission',
                },
                5: {
                    1: 'Food & Drug Administration',
                    2: 'General Services Administration',
                    3: 'Government Accounting Office (GAO)',
                    4: 'Health Care Financing Administration',
                    5: 'Immigration & Naturalization Service (includes border patrol)',
                    6: 'Internal Revenue Service (IRS)',
                    7: 'Interstate Commerce Commission',
                    8: 'Merit Systems Protection Board',
                    9: 'National Credit Union Association',
                    10: 'National Labor Relations Board',
                    11: 'Nuclear Regulatory Commission',
                },
                6: {
                    1: 'Occupational Safety & Health Administration',
                    2: 'Occupational Safety & Health Review Commission',
                    3: 'Office of the Federal Inspector',
                    4: 'Office of Management & Budget',
                    5: 'Office of Personnel Management',
                    6: 'Office of Workers Compensation Program',
                    7: 'Parole board or parole commisssion, or prison official, or US Bureau of Prisons',
                    8: 'Patent Office',
                    9: 'Postal Rate Commission (U.S.)',
                    10: 'Postal Service (U.S.)',
                    11: 'RR Adjustment Board',
                    12: 'RR Retirement Board',
                },
                7: {
                    1: 'Securities & Exchange Commission',
                    2: 'Small Business Administration',
                    3: 'Veterans Administration',
                },
                8: {
                    1: 'DC in its corporate capacity',
                    2: 'legislative body for DC local government',
                    3: 'mayor, agency head or top administrator',
                    4: 'bureaucracy providing service',
                    5: 'bureaucracy in charge of regulation',
                    6: 'bureaucracy in charge of general administration',
                    7: 'judicial',
                    8: 'other',
                },
                9: {
                    1: 'United States - in corporate capacity (i.e., as representative of "the people") - in criminal cases',
                    2: 'United States - in corporate capacity - civil cases',
                    3: 'special wartime agency',
                    4: 'Other unlisted federal agency (includes the President of the US)',
                    5: 'Unclear or nature not ascertainable',
                }
            }
        },
    },
    4: {
        2: {
            'instruction': 'Your task is to determine which category of substate government best describes this litigant.',
            'question': 'Which category of substate government best describes this litigant?',
            'answer_choices': {
                1: 'legislative',
                2: 'executive/administrative',
                3: 'bureaucracy providing services',
                4: 'bureaucracy in charge of regulation',
                5: 'bureaucracy in charge of general administration',
                6: 'judicial',
                7: 'other',
            },
        },
        3: {
            'instruction': 'Your task is to determine which specific substate government agency best describes this litigant.',
            'question': 'Which specific substate government agency best describes this litigant?',
            'choosing_rule': lambda x: str(x)[1],  # second digit
            'possible_choices': {
                1: {
                    1: 'City/county council',
                    2: 'School Board, board of trustees for college or junior college',
                    3: 'Other legislative body',
                    0: 'not ascertained',
                },
                2: {
                    1: 'CEO or officials in charge of agency',
                    2: 'Mayor/county executive',
                    3: 'Primary or secondary school system CEO',
                    4: 'Other CEO or administrative official (except prison)',
                    0: 'not ascertained',
                },
                3: {
                    1: 'Police, Sheriff',
                    2: 'Fire',
                    3: 'Taxation',
                    4: 'Human Services/Welfare/Health Care',
                    5: 'Streets and Highways',
                    6: 'Transportation',
                    7: 'Election Processes',
                    8: 'Education - Not School Board',
                    9: 'Other Service Activity',
                    0: 'not ascertained',
                },
                4: {
                    1: 'Environment',
                    2: 'Market Practices',
                    3: 'Transportation',
                    4: 'Professions (licensing)',
                    5: 'Labor-Management',
                    6: 'Communications',
                    7: 'Zoning/Land Use',
                    8: 'Building and Housing',
                    9: 'Other Regulating Activity',
                    0: 'not ascertained',
                },
                5: {
                    1: 'Personnel',
                    2: 'Other General Administration',
                    0: 'not ascertained',
                },
                6: {
                    1: 'Judge or Court (local trial court judge or justice of peace)',
                    2: 'Prosecutor/district attorney',
                    3: 'Jail/Prison/Probation Official and Organization (includes prison hospitals; includes juvenile correction officials)',
                    4: 'Other Judical Official',
                    0: 'not ascertained',
                },
                7: {
                    1: 'City of, county of, etc. - in corporate capacity - criminal case',
                    2: 'city of, county of, etc. - in corporate capacity - civil case',
                    3: 'Other sub-state activity',
                    0: 'not ascertained',
                }
            }
        },
    },
    5: {
        2: {
            'instruction': 'Your task is to determine which category of state government best describes this litigant.',
            'question': 'Which category of state government best describes this litigant?',
            'answer_choices': {
                1: 'legislative',
                2: 'executive/administrative',
                3: 'bureaucracy providing services',
                4: 'bureaucracy in charge of regulation',
                5: 'bureaucracy in charge of general administration',
                6: 'judicial',
                7: 'other',
            },
        },
        3: {
            'instruction': 'Your task is to determine which specific state government agency best describes this litigant.',
            'question': 'Which specific state government agency best describes this litigant?',
            'choosing_rule': lambda x: str(x)[1],  # second digit
            'possible_choices': {
                1: {
                    1: 'Legislature or separate house as an organization',
                    2: 'Legislative Committee or Commission',
                    3: 'Other Legislative Unit',
                    0: 'not ascertained',
                },
                2: {
                    1: 'Governor',
                    2: 'Attorney General',
                    3: 'Secretary of State',
                    4: 'Other Administrative Officer NOT detailed below',
                    0: 'not ascertained',
                },
                3: {
                    1: 'Police',
                    2: 'Fire',
                    3: 'Taxation',
                    4: 'Human Services/Welfare/Health Care',
                    5: 'Streets and Highways',
                    6: 'Transportation',
                    7: 'Election processes',
                    8: 'Education',
                    9: 'Other Service Activity',
                    0: 'not ascertained',
                },
                4: {
                    1: 'Environment',
                    2: 'Market Practices',
                    3: 'Transportation',
                    4: 'Professions (licensing)',
                    5: 'Labor-Management',
                    6: 'Communications',
                    7: 'Zoning/Land Use',
                    8: 'Building and Housing',
                    9: 'Other Regulating Activity',
                    0: 'not ascertained',
                },
                5: {
                    1: 'Personnel',
                    2: 'Other General Administration',
                    0: 'not ascertained',
                },
                6: {
                    1: 'Judge (non-local judge; appellate judge)',
                    2: 'Prosecutor/district attorney (non-local, e.g., special prosecutor)',
                    3: 'Jail/Prison/Probation Official (includes juvenile officials)',
                    4: 'Other judicial official',
                    0: 'not ascertained',
                },
                7: {
                    1: 'state of ___ - state in its corporate capacity in criminal cases',
                    2: 'state 0f ___ - state in its corporate capacity in civil cases',
                    3: 'other state level activity',
                    0: 'not ascertained',
                },
            }
        },

    },
    7: {
        2: {
            'instruction': 'Your task is to determine the gender of this litigant. ' \
                'Use names to classify the party\'s sex only if there is little ambiguity ' \
                '(e.g., the sex of "Chris" should be coded as "not ascertained").',
            'question': 'What is the gender of this litigant?' \
                'Use names to classify the party\'s sex only if there is little ambiguity.',
            'answer_choices': {
                0: 'not ascertained',
                1: 'male - indication in opinion (e.g., use of masculine pronoun)',
                2: 'male - assumed because of name',
                3: 'female - indication in opinion of gender',
                4: 'female - assumed because of name',
            },
        },
        3: {
            'instruction': 'Your task is to determine the race or ethnic identity of this litigant as identified in the opinion. ' \
                'Names may be used to classify a person as hispanic if there is little ambiguity. ' \
                'All aliens are coded as "not ascertained".',
            'question': 'What is the race or ethnic identity of this litigant as identified in the opinion?',
            'answer_choices': {
                0: 'not ascertained',
                1: 'caucasian - specific indication in opinion',
                2: 'black - specific indication in opinion',
                3: 'native american - specific indication in opinion',
                4: 'native american - assumed from name',
                5: 'asian - specific indication in opinion',
                6: 'asian - assumed from name',
                7: 'hispanic - specific indication in opinion',
                8: 'hispanic - assumed from name',
                9: 'other',
            },
        },
        4: {
            'instruction': 'Your task is to determine the citizenship of this litigant as indicated in the opinion.',
            'question': 'What is the citizenship of this litigant as indicated in the opinion?',
            'answer_choices': {
                0: 'not ascertained',
                1: 'US citizen',
                2: 'alien',
            },
        },
        5: {
            'instruction': 'Your task is to determine which of these categories best describes the income of the litigant. ' \
                'Consider the following categories: "not ascertained", ' \
                '"poor + wards of state" (e.g., patients at state mental hospital; not prisoner unless specific indication that poor), ' \
                '"presumed poor" (e.g., migrant farm worker), ' \
                '"presumed wealthy" (e.g., high status job - like medical doctors, executives of corporations that are national ' \
                'in scope, professional athletes in the NBA or NFL; upper 1/5 of income bracket), ' \
                '"clear indication of wealth in opinion", ' \
                '"other - above poverty line but not clearly wealthy" (e.g., public school teachers, federal government employees)." ' \
                'Note that "poor" means below the federal poverty line; e.g., welfare or food stamp recipients. ' \
                'There must be some specific indication in the opinion that you can point to before anyone is classified anything other than "not ascertained". ' \
                'Prisoners filing "pro se" were classified as poor, but litigants in civil cases who proceed pro se were not presumed to be poor. ' \
                'Wealth obtained from the crime at issue in a criminal case was not counted when determining the wealth of the criminal defendant (e.g., drug dealers).',
            'question': 'Which of these categories best describes the income of the litigant?',
            'answer_choices': {
                0: 'not ascertained',
                1: 'poor + wards of state',
                2: 'presumed poor',
                3: 'presumed wealthy',
                4: 'clear indication of wealth in opinion',
                5: 'other - above poverty line but not clearly wealthy',
            },
        },
    },
    8: {
        2: {
            'instruction': 'Your task is to determine which of the following categories best describes the litigant.',
            'question': 'Which of the following categories best describes the litigant?',
            'answer_choices': {
                1: 'fiduciary, executor, or trustee',
                2: 'other',
                3: 'nature of the litigant not ascertained',
            },
        },
        3: {
            'instruction': 'Your task is to determine which of the following specific subcategories best describes the litigant.',
            'question': 'Which of the following specific subcategories best describes the litigant?',
            'choosing_rule': lambda x: str(x)[1],  # second digit
            'possible_choices': {
                1: {
                    1: 'trustee in bankruptcy - institution',
                    2: 'trustee in bankruptcy - individual',
                    3: 'executor or administrator of estate - institution',
                    4: 'executor or administrator of estate - individual',
                    5: 'trustees of private and charitable trusts - institution',
                    6: 'trustee of private and charitable trust - individual',
                    7: 'conservators, guardians and court appointed trustees for minors, mentally incompetent',
                    8: 'other fiduciary or trustee',
                    0: 'specific subcategory not ascertained',
                },
                2: {
                    1: 'Indian Tribes',
                    2: 'Foreign Government',
                    3: 'Multi-state agencies, boards, etc. (e.g., Port Authority of NY)',
                    4: 'International Organizations',
                    5: 'Other',
                    0: 'Not ascertained',
                },
            }
        },
    },
}

litigant_general_categories = {
    1: 'private business (including criminal enterprises)',
    2: 'private organization or association',
    3: 'federal government (including DC)',
    4: 'sub-state government (e.g., county, local, special district)',
    5: 'state government (includes territories & commonwealths)',
    6: 'government - level not ascertained',
    7: 'natural person (excludes persons named in their official ' \
            'capacity or who appear because of a role in a private organization)',
    8: 'miscellaneous',
    9: 'not ascertained',
}

def build_app_resp_tasks():
    all_header = f'{header}\n{header_participants}\n{header_nature_participants}\n'

    litigant_text = {
        'appel1': 'first listed appellant',
        'appel2': 'second listed appellant',
        'respond1': 'first listed respondent',
        'respond2': 'second listed respondent',
    }

    for lit_code, lit_txt in litigant_text.items():
        for cat_code, cat_txt in litigant_general_categories.items():
            if cat_code in [6, 9]:
                continue
            cat_header = f'The nature of this litigant falls into the category "{cat_txt}"'
            for digit, task in party_details[cat_code].items():
                task_name = f"songer_{lit_code}_{cat_code}_{digit}"
                q_header = f'This question concerns the {lit_txt}. {cat_header}'
                task_header = f'{all_header}\nYour task concerns the {lit_txt}. {cat_header}'

                if 'answer_choices' in task:
                    task_ = {
                        'name': task_name,
                        'instruction': f'{task_header}. {task["instruction"]}',
                        'question': f'{q_header}. {task["question"]}',
                        'answer_choices': task['answer_choices'],
                    }
                    yield lit_code, cat_code, digit, None, task_
                else:
                    assert 'possible_choices' in task, 'possible_choices must be provided for task without answer_choices'
                    for choice_code, answer_choices in task['possible_choices'].items():
                        digit_header = party_details[cat_code][digit-1]['answer_choices'][choice_code]
                        task_ = {
                            'name': f"{task_name}_{choice_code}",
                            'instruction': f'{task_header}, specifically "{digit_header}". {task["instruction"]}',
                            'question': f'{q_header}, specifically "{digit_header}". {task["question"]}',
                            'answer_choices': answer_choices,
                        }
                
                        yield lit_code, cat_code, digit, choice_code, task_

tasks_participants = {
    'numappel' : {
        'name': 'songer_numappel',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n' \
            'Your specific task is to determine the total number of appellants in the case. ' \
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case? Answer with a number.',
        'type': 'int',
    },
    'appnatpr': {
        'name': 'songer_appnatpr',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "natural persons". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "natural persons"? Answer with a number.',
        'type': 'int',
    },
    'appbus': {
        'name': 'songer_appbus',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "private business and its executives". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "private business and its executives"? Answer with a number.',
        'type': 'int',
    },
    'appnonp': {
        'name': 'songer_appnonp',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "groups and associations". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "groups and associations"? Answer with a number.',
        'type': 'int',
    },
    'appfed': {
        'name': 'songer_appfed',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "the federal government, its agencies, and officials". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "the federal government, its agencies, and officialss"? Answer with a number.',
        'type': 'int',
    },
    'appsubst': {
        'name': 'songer_appsubst',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "sub-state governments, their agencies, and officials". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "sub-state governments, their agencies, and officials"? Answer with a number.',
        'type': 'int',
    },
    'appstate': {
        'name': 'songer_appstate',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "state governments, their agencies, and officials". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "state governments, their agencies, and officials"? Answer with a number.',
        'type': 'int',
    },
    'appfiduc': {
        'name': 'songer_appfiduc',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of appellants in the case ' \
            'that fall into the category "fiduciaries". '
            'If the total number cannot be determined (e.g., if the appellant is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of appellants in the case ' \
            'that fall into the category "fiduciaries"? Answer with a number.',
        'type': 'int',
    },
    'ap_stid': {
        'name': 'songer_app_stid',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n' \
            'Your task is to identify the state of the first listed state or local government agency that is an appellant.',
        'question': 'What is the state of the first listed state or local government agency that is an appellant?',
        'answer_choices': states_file,
    },
    'bank_ap1': {
        'name': 'songer_bank_app1',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether or not the first listed appellant is bankrupt. ' \
            'If there is no indication of whether or not the appellant is bankrupt, the appellant ' \
            'is presumed to be not bankrupt.',
        'question': 'Is the first listed appellant bankrupt?',
        'answer_choices': {
            1: 'Yes',
            2: 'No',
        }
    },
    'genapel1': {
        'name': 'songer_genapel1',
        'instruction': f'{header}\n{header_participants}\n{header_nature_participants}\n' \
            'Your task is to determine the nature of the first listed appellant.',
        'question': 'What is the nature of the first listed appellant?',
        'answer_choices': litigant_general_categories,
    },
    'genapel2': {
        'name': 'songer_genapel2',
        'instruction': f'{header}\n{header_participants}\n{header_nature_participants}\n' \
            'Your task is to determine the nature of the second listed appellant. ' \
            'If there are more than two appellants and at least one of the additional appellants ' \
            'has a different general category from the first appellant, ' \
            'then consider the first appellant with a different general category to be the second appellant.',
        'question': 'What is the nature of the second listed appellant ' \
            'whose detailed code is not identical to the code for the first listed appellant?',
        'answer_choices': litigant_general_categories,
    },
    'bank_ap2': {
        'name': 'songer_bank_app2',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether or not the second listed appellant is bankrupt. ' \
            'If there is no indication of whether or not the appellant is bankrupt, the appellant ' \
            'is presumed to be not bankrupt.',
        'question': 'Is the second listed appellant bankrupt?',
        'answer_choices': {
            1: 'Yes',
            2: 'No',
        }
    },
    'realapp': {
        'name': 'songer_realapp',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether or not the formally listed appellants ' \
            'in the case are the "real parties." That is, are they the parties whose ' \
            'real interests are most directly at stake? (e.g., in some appeals ' \
            'of adverse habeas corpus petition decisions, the respondent is ' \
            'listed as the judge who denied the petition, but the real parties ' \
            'are the prisoner and the warden of the prison) (another example ' \
            'would be "Jones v A 1990 Rolls Royce" where Jones is a drug agent ' \
            'trying to seize a car which was transporting drugs - the real party ' \
            'would be the owner of the car). ' \
            'For cases in which an independent regulatory agency is the ' \
            'listed appellant, the following rule was adopted: If the agency ' \
            'initiated the action to enforce a federal rule or the agency was ' \
            'sued by a litigant contesting an agency action, then the agency was ' \
            'coded as a real party. However, if the agency initially only acted ' \
            'as a forum to settle a dispute between two other litigants, and the ' \
            'agency is only listed as a party because its ruling in that dispute ' \
            'is at issue, then the agency is considered not to be a real party. ' \
            'For example, if a union files an unfair labor practices charge ' \
            'against a corporation, the NLRB hears the dispute and rules for the ' \
            'union, and then the NLRB petitions the court of appeals for ' \
            'enforcement of its ruling in an appeal entitled "NLRB v Widget ' \
            'Manufacturing, INC." the NLRB would be coded as not a real party. ' \
            'Note that under these definitions, trustees are usually "real ' \
            'parties" and parents suing on behalf of their children and a spouse ' \
            'suing on behalf of their injured or dead spouse are also "real ' \
            'parties."',
        'question': 'Are the formally listed appellants in the case the "real parties", that is, ' \
            'are they the parties whose real interests are most directly at stake?',
        'answer_choices': {
            0: 'both 1st and 2nd listed appellants are real parties (or only one appellant, and that appellant is a real party)',
            1: 'the 1st appellant is not a real party',
            2: 'the 2nd appellant is not a real party',
            3: 'neither the 1st nor the 2nd appellants are real parties',
            4: 'not ascertained',
        },
    },
    'numresp': {
        'name': 'songer_numresp',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n' \
            'Your specific task is to determine the total number of respondents in the case. ' \
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case? Answer with a number.',
        'type': 'int',
    },
    'r_natpr': {
        'name': 'songer_r_natpr',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "natural persons". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "natural persons"? Answer with a number.',
        'type': 'int',
    },
    'r_bus': {
        'name': 'songer_r_bus',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "private business and its executives". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "private business and its executives"? Answer with a number.',
        'type': 'int',
    },
    'r_nonp': {
        'name': 'songer_r_nonp',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "groups and associations". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "groups and associations"? Answer with a number.',
        'type': 'int',
    },
    'r_fed': {
        'name': 'songer_r_fed',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "the federal government, its agencies, and officials". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "the federal government, its agencies, and officialss"? Answer with a number.',
        'type': 'int',
    },
    'r_subst': {
        'name': 'songer_r_subst',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "sub-state governments, their agencies, and officials". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "sub-state governments, their agencies, and officials"? Answer with a number.',
        'type': 'int',
    },
    'r_state': {
        'name': 'songer_r_state',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "state governments, their agencies, and officials". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "state governments, their agencies, and officials"? Answer with a number.',
        'type': 'int',
    },
    'r_fiduc': {
        'name': 'songer_r_fiduc',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_specific_app}\n' \
            'Your specific task is to determine the total number of respondents in the case ' \
            'that fall into the category "fiduciaries". '
            'If the total number cannot be determined (e.g., if the respondent is ' \
            'listed as "Smith, et. al." and the opinion does not specify who is ' \
            'included in the "et.al."), then answer 99.',
        'question': 'What is the total number of respondents in the case ' \
            'that fall into the category "fiduciaries"? Answer with a number.',
        'type': 'int',
    },
    'r_stid': {
        'name': 'songer_r_stid',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n' \
            'Your task is to identify the state of the first listed state or local government agency that is a respondent.',
        'question': 'What is the state of the first listed state or local government agency that is a respondent?',
        'answer_choices': states_file,
    },
    'genresp1': {
        'name': 'songer_genresp1',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_nature_participants}\n' \
            'Your task is to determine the nature of the first listed respondent.',
        'question': 'What is the nature of the first listed respondent?',
        'answer_choices': litigant_general_categories,
    },
    'bank_r1': {
        'name': 'songer_bank_r1',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether or not the first listed respondent is bankrupt. ' \
            'If there is no indication of whether or not the respondent is bankrupt, the respondent ' \
            'is presumed to be not bankrupt.',
        'question': 'Is the first listed respondent bankrupt?',
        'answer_choices': {
            1: 'Yes',
            2: 'No',
        },
    },
    'genresp2': {
        'name': 'songer_genresp2',
        'instruction': f'{header}\n{header_participants}\n{head_appellants}\n{header_nature_participants}\n' \
            'Your task is to determine the nature of the second listed respondent. ' \
            'If there are more than two respondents and at least one of the additional respondents ' \
            'has a different general category from the first respondent, ' \
            'then consider the first respondent with a different general category to be the second respondent.',
        'question': 'What is the nature of the second listed respondent ' \
            'whose detailed code is not identical to the code for the first listed respondent?',
        'answer_choices': litigant_general_categories,
    },
    'bank_r2': {
        'name': 'songer_bank_r2',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether or not the second listed respondent is bankrupt. ' \
            'If there is no indication of whether or not the respondent is bankrupt, the respondent ' \
            'is presumed to be not bankrupt.',
        'question': 'Is the second listed respondent bankrupt?',
        'answer_choices': {
            1: 'Yes',
            2: 'No',
        }
    },
    'realresp': {
        'name': 'songer_realresp',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether or not the formally listed respondents ' \
            'in the case are the "real parties." That is, are they the parties whose ' \
            'real interests are most directly at stake? (e.g., in some appeals ' \
            'of adverse habeas corpus petition decisions, the respondent is ' \
            'listed as the judge who denied the petition, but the real parties ' \
            'are the prisoner and the warden of the prison) (another example ' \
            'would be "Jones v A 1990 Rolls Royce" where Jones is a drug agent ' \
            'trying to seize a car which was transporting drugs - the real party ' \
            'would be the owner of the car). ' \
            'For cases in which an independent regulatory agency is the ' \
            'listed respondent, the following rule was adopted: If the agency ' \
            'initiated the action to enforce a federal rule or the agency was ' \
            'sued by a litigant contesting an agency action, then the agency was ' \
            'coded as a real party. However, if the agency initially only acted ' \
            'as a forum to settle a dispute between two other litigants, and the ' \
            'agency is only listed as a party because its ruling in that dispute ' \
            'is at issue, then the agency is considered not to be a real party. ' \
            'For example, if a union files an unfair labor practices charge ' \
            'against a corporation, the NLRB hears the dispute and rules for the ' \
            'union, and then the NLRB petitions the court of appeals for ' \
            'enforcement of its ruling in an appeal entitled "NLRB v Widget ' \
            'Manufacturing, INC." the NLRB would be coded as not a real party. ' \
            'Note that under these definitions, trustees are usually "real ' \
            'parties" and parents suing on behalf of their children and a spouse ' \
            'suing on behalf of their injured or dead spouse are also "real ' \
            'parties."',
        'question': 'Are the formally listed respondents in the case the "real parties", that is, ' \
            'are they the parties whose real interests are most directly at stake?',
        'answer_choices': {
            0: 'both 1st and 2nd listed respondents are real parties (or only one respondent, and that respondent is a real party)',
            1: 'the 1st respondent is not a real party',
            2: 'the 2nd respondent is not a real party',
            3: 'neither the 1st nor the 2nd respondents are real parties',
            4: 'not ascertained',
        },
    },
    'counsel1': {
        'name': 'songer_counsel1',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine the nature of the counsel for the appellant. ' \
            'If name of attorney was given with no other indication of affiliation, ' \
            'assume it is private - unless a government agency was the party',
        'question': 'What is the nature of the counsel for the appellant?',
        'answer_choices': {
            1: 'none (pro se)',
            2: 'court appointed',
            3: 'legal aid or public defender',
            4: 'private',
            5: 'government - US',
            6: 'government - state or local',
            7: 'interest group, union, professional group',
            8: 'other or not ascertained',
        },
    },
    'counsel2': {
        'name': 'songer_counsel2',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine the nature of the counsel for the respondent. ' \
            'If name of attorney was given with no other indication of affiliation, ' \
            'assume it is private - unless a government agency was the party',
        'question': 'What is the nature of the counsel for the respondent?',
        'answer_choices': {
            1: 'none (pro se)',
            2: 'court appointed',
            3: 'legal aid or public defender',
            4: 'private',
            5: 'government - US',
            6: 'government - state or local',
            7: 'interest group, union, professional group',
            8: 'other or not ascertained',
        },
    },
    'amicus': {
        'name': 'songer_amicus',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine or not there was any amicus participation before the court of appeals.',
        'question': 'Was there any amicus participation before the court of appeals?',
        'answer_choices': {
            0: 'no amicus participation on either side',
            1: '1 separate amicus brief was filed',
            2: '2 separate amicus briefs were filed',
            3: '3 separate amicus briefs were filed',
            4: '4 separate amicus briefs were filed',
            5: '5 separate amicus briefs were filed',
            6: '6 separate amicus briefs were filed',
            7: '7 separate amicus briefs were filed',
            8: '8 or more separate amicus briefs were filed',
            9: 'not ascertained',
        },
    },
    'interven': {
        'name': 'songer_interven',
        'instruction': f'{header}\n{header_participants}\n' \
            'Your task is to determine whether one or more individuals or groups ' \
            'sought to formally intervene in the appeals court consideration of the case.',
        'question': 'Did one or more individuals or groups seek to formally intervene in the appeals court consideration of the case?',
        'answer_choices': {
            0: 'no intervenor in case',
            1: 'intervenor = appellant',
            2: 'intervenor = respondent',
            3: 'yes, both appellant & respondent',
            9: 'not applicable',
        },
    },
}

header_issue = 'Your task is to identify the issue in the case, that is, the social and/or political ' \
    'context of the litigation in which more purely legal issues are argued. ' \
    'Put somewhat differently, this field identifies the nature of the conflict between the litigants. ' \
    'The focus here is on the subject matter of the controversy rather than its legal basis.'

case_issues = {
    1: {
        2: {
            1: 'federal offense',
            2: 'state offense',
            3: 'not determined whether state or federal offense',
        },
        3: {
            1: {
                101: 'murder',
                102: 'rape',
                103: 'arson',
                104: 'aggravated assault',
                105: 'robbery',
                106: 'burglary',
                107: 'auto theft',
                108: 'larceny (over $50)',
                109: 'other violent crimes',
                110: 'narcotics',
                111: 'alcohol related crimes, prohibition',
                112: 'tax fraud',
                113: 'firearm violations',
                114: 'morals charges (e.g., gambling, prostitution, obscenity)',
                115: 'criminal violations of government regulations of business',
                116: 'other white collar crime (involving no force or threat of force; e.g., embezzlement, computer fraud,bribery)',
                117: 'other crimes',
                118: 'federal offense, but specific crime not ascertained'
            },
            2: {
                121: 'murder',
                122: 'rape',
                123: 'arson',
                124: 'aggravated assault',
                125: 'robbery',
                126: 'burglary',
                127: 'auto theft',
                128: 'larceny (over $50)',
                129: 'other violent crimes',
                130: 'narcotics',
                131: 'alcohol related crimes, prohibition',
                132: 'tax fraud',
                133: 'firearm violations',
                134: 'morals charges (e.g., gambling, prostitution, obscenity)',
                135: 'criminal violations of government regulations of business',
                136: 'other white collar crime (involving no force or threat of force; e.g., embezzlement, computer fraud,bribery)',
                137: 'other state crimes',
                138: 'state offense, but specific crime not ascertained'
            },
            3: {
                141: 'murder',
                142: 'rape',
                143: 'arson',
                144: 'aggravated assault',
                145: 'robbery',
                146: 'burglary',
                147: 'auto theft',
                148: 'larceny (over $50)',
                149: 'other violent crimes',
                150: 'narcotics',
                151: 'alcohol related crimes, prohibition',
                152: 'tax fraud',
                153: 'firearm violations',
                154: 'morals charges (e.g., gambling, prostitution, obscenity)',
                155: 'criminal violations of government regulations of business',
                156: 'other white collar crime (involving no force or threat of force; e.g., embezzlement, computer fraud,bribery)',
                157: 'other crimes',
                158: 'specific crime not ascertained'
            }
        },
    },
    2: {
        2: {
            1: 'civil rights claims by prisoners and those accused of crimes',
            2: 'voting rights, race discrimination, sex discrimination',
            3: 'other civil rights',
        },
        3: {
            1: {
                201: 'suit for damages for false arrest or false confinement',
                202: 'cruel and unusual punishment',
                203: 'due process rights in prison',
                204: 'denial of other rights of prisoners - 42 USC 1983 suits',
                205: 'denial or revocation of parole - due process grounds',
                206: 'other denial or revocation of parole',
                207: 'other prisoner petitions',
                208: 'excessive force used in arrest',
                209: 'other civil rights violations alleged by criminal defendants',
            },
            2: {
                210: 'voting rights - reapportionment & districting',
                211: 'participation rights - rights of candidates or groups to fully participate in the political process; access to ballot',
                212: 'voting rights - other (includes race discrimination in voting)',
                213: 'desegregation of schools',
                214: 'other desegregation',
                221: 'employment race discrimination - alleged by minority',
                222: 'other race discrimination - alleged by minority',
                223: 'employment: race discrimination - alleged by caucasin (or opposition to affirmative action plan which benefits minority)',
                224: 'other reverse race discrimination claims',
                231: 'employment: sex discrimination - alleged by woman',
                232: 'pregnancy discrimination',
                233: 'other sex discrimination - alleged by woman',
                234: 'employment: sex discrimination - alleged by man (or opposition to affirmative action plan which benefits women)',
                235: 'other sex discrimination - alleged by man',
                239: 'suits raising 42 USC 1983 claims based on race or sex discrimination',
            },
            3: {
                241: 'alien petitions - (includes disputes over attempts at deportation)',
                251: 'indian rights and law',
                261: 'juveniles',
                271: 'poverty law, rights of indigents (civil)',
                281: 'rights of handicapped (includes employment)',
                282: 'age discrimination (includes employment)',
                283: 'discrimination based on religion or nationality',
                284: 'discrimination based on sexual preference federal government (other than categories above)',
                291: 'other 14th amendment and civil rights act cases',
                290: '290 challenge to hiring, firing, promotion decision of federal government (other than categories above)',
                299: 'other civil rights',
            }
        }
    },
    3: {
        2: {
            1: 'religion, press, commercial',
            2: 'speech and other expression',
        },
        3: {
            1: {
                301: 'commercial speech',
                302: 'libel, slander, defamation',
                303: 'free exercise of religion',
                304: 'establishment of religion (other than aid to parochial schools)',
                305: 'aid to parochial schools',
                306: 'press',
            },
            2: {
                307: 'obscenity',
                308: 'association',
                309: 'federal internal security and communist control acts, loyalty oaths, security risks',
                310: 'legality of expression in context of overt acts (speeches, parades, picketing, etc.) protesting race discrimination',
                311: 'overt acts - opposition to war and the military',
                312: 'conscientious objection to military service or other first amendment challenges to the military',
                313: 'expression of political or social beliefs conflicting with regulation of physical activity (includes demonstrations, parades, canvassing, picketing)',
                314: 'threats to peace, safety ,and order (except those covered above) (includes fighting words, clear and present danger, incitement to riot)',
                315: 'challenges to campaign spending limits or other limits on expression in political campaigns',
                399: 'other (includes tests of belief)',
            },
        },
    },
    4: {
        3: {
            410: 'denial of fair hearing or notice - government employees (includes claims of terminated government workers)',
            411: 'denial of hearing or notice in non-employment context',
            412: 'taking clause (i.e., denial of due process under the "taking" clause of the 5th or 14th Amendments)',
            413: 'freedom of information act and other claims of rights of access (includes all cases involving dispute over requests for information even if it does not involve the freedom of information act)',
            499: 'other due process issues',
        },
    },
    5: {
        3: {
            501: 'abortion rights',
            502: 'homosexual rights where privacy claim raised',
            503: 'contraception and other privacy claims related to marital relations or sexual behavior (not in 501 or 502)',
            504: 'suits demanding compensation for violation of privacy rights (e.g., 1983 suits)',
            505: 'mandatory testing (for drugs, AIDs, etc)',
            506: 'mandatory sterilization',
            507: 'right to die or right to refuse medical help',
            599: 'other',
        },
    },
    6: {
        3: {
            601: 'union organizing',
            602: 'unfair labor practices',
            603: 'Fair Labor Standards Act issues',
            604: 'Occupational Safety and Health Act issues (including OSHA enforcement)',
            605: 'collective bargaining',
            606: 'conditions of employment',
            607: 'employment of aliens',
            608: 'which union has a right to represent workers',
            609: 'non civil rights grievances by worker against union (e.g., union did not adequately represent individual)',
            610: 'other labor relations',
        },
    },
    7: {
        2: {
            1: 'taxes, patents, copyright',
            2: 'torts',
            3: 'commercial disputes',
            4: 'bankruptcy, antitrust, securities',
            5: 'misc economic regulation and benefits',
            6: 'property disputes',
            7: 'other',
        },
        3: {
            1: {
                701: 'state or local tax',
                702: 'federal taxation - individual income tax (includes taxes of individuals, fiduciaries, & estates)',
                703: 'federal tax - business income tax (includes corporate and parnership)',
                704: 'federal tax - excess profits',
                705: 'federal estate and gift tax',
                706: 'federal tax - other',
                710: 'patents',
                711: 'copyrights',
                712: 'trademarks',
                713: 'trade secrets, personal intellectual property',
            },
            2: {
                720: 'motor vehicle',
                721: 'airplane',
                722: 'product liability',
                723: 'federal employer liability; injuries to dockworkers and longshoremen',
                724: 'other government tort liability',
                725: 'workers compensation',
                726: 'medical malpractice',
                727: 'other personal injury',
                728: 'fraud',
                729: 'other property damage',
                730: 'other torts',
            },
            3: {
                731: 'contract disputes-general (private parties) (includes breach of contract, disputes over meaning of contracts, suits for specific performance, disputes over whether contract fulfilled, claims that money owed on contract) (Note: this category is not used when the dispute fits one of the more specific categories below)',
                732: 'disputes over government contracts',
                733: 'insurance disputes',
                734: 'debt collection, disputes over loans',
                735: 'consumer disputes with retail business or providers of services',
                736: 'breach of fiduciary duty; disputes over franchise agreements',
                737: 'contract disputes - was there a contract, was it a valid contract ?',
                738: 'commerce clause challenges to state or local government action',
                739: 'other contract disputes- (includes misrepresentation or deception in contract, disputes among contractors or contractors and subcontractors, indemnification claims)',
                740: 'private economic disputes (other than contract disputes)',
            },
            4: {
                741: 'bankruptcy - private individual (e.g., chapter 7)',
                742: 'bankruptcy - business reorganization (e.g., chapter 11)',
                743: 'other bankruptcy',
                744: 'antitrust - brought by individual or private business (includes Clayton Act; Sherman Act; and Wright-Patman)',
                745: 'antitrust - brought by government',
                746: 'regulation of, or opposition to mergers on other than anti-trust grounds',
                747: 'securities - conflicts between private parties (including corporations)',
                748: 'government regulation of securities',
            },
            5: {
                750: 'social security benefits (including SS disability payments)',
                751: 'other government benefit programs (e.g., welfare, RR retirement, veterans benefits, war risk insurance, food stamps)',
                752: 'state or local economic regulation',
                753: 'federal environmental regulation',
                754: 'federal consumer protection regulation (includes pure food and drug, false advertising)',
                755: 'rent control; excessive profits; government price controls',
                756: 'federal regulation of transportation',
                757: 'oil, gas, and mineral regulation by federal government',
                758: 'federal regulation of utilities (includes telephone, radio, TV, power generation)',
                759: 'other commercial regulation (e.g.,agriculture, independent regulatory agencies) by federal government',
                760: 'civil RICO suits',
                761: 'admiralty - personal injury (note:suits against government under admiralty should be classified under the government tort category above)',
                762: 'admiralty - seamens wage disputes',
                763: 'admiralty - maritime contracts, charter contracts',
                764: 'admiralty other',
            },
            6: {
                770: 'disputes over real property (private)',
                771: 'eminent domain and disputes with government over real property',
                772: 'landlord - tenant disputes',
                773: 'government seizure of property - as part of enforcement of criminal statutes',
                774: 'government seizure of property - civil (e.g., for deliquent taxes, liens)',
            },
            7: {
                799: 'other economic activity',
            },
        },
    },
    9: {
        3: {
            901: 'miscellaneous interstate conflict',
            902: 'other federalism issue (only code as issue if opinion explicitly discusses federalism as an important issue - or if opinion explicity discusses conflict of state power vs federal power)',
            903: 'attorneys (disbarment; etc)',
            904: 'selective service or draft issues (which do not include 1st amendment challenges)',
            905: 'challenge to authority of magistrates, special masters, etc.',
            906: 'challenge to authority of bankruptcy judge or referees in bankruptcy',
            910: 'Indian law - criminal verdict challenged due to interpretation of tribal statutes or other indian law',
            911: 'Indian law - commercial disputes based on interpretation of Indian treaties or law (includes disputes over mineral rights)',
            912: 'Indian law - Indian claims acts and disputes over real property (includes Alaska Native Claims Act)',
            913: 'Indian law - federal regulation of Indian land and affairs',
            914: 'Indian law - state/local authority over Indian land and affairs',
            915: 'Indian law - tribal regulation of economic activities (includes tribal taxation)',
            916: 'other Indian law',
            920: 'international law',
            921: 'immigration (except civil rights claims of immigrants and aliens)',
            999: 'other',
            000: 'not ascertained',
        },
    },
}

direct = {
    'instruction': f'{header}\n' \
        'Your task is to determine the ideological directionality of the court of appeals decision, coded ' \
        'as "liberal" or "conservative".{lib_desc} ' \
        'Consider the directionality to be "mixed" if the ' \
        'directionality of the decision was intermediate to the extremes defined above or if the decision was mixed (e.g., the conviction of ' \
        'defendant in a criminal trial was affirmed on one count but reversed on a second count or if the conviction was afirmed but the sentence was reduced). ' \
        'Consider "not ascertained" if the directionality could not be determined or if the outcome could not be classified according to any conventional outcome standards.',
    'question': 'What is the ideological directionality of the court of appeals decision?',
    'answer_choices': {
        1: 'conservative',
        3: 'liberal',
        2: 'mixed',
        0: 'not ascertained',
    },
    'fill_in': ['lib_desc'],
    'lib_desc': {
        1: ' Consider liberal to be  for the defendant.',
        2: ' Consider liberal to be for the position of the ' \
            'prisoner; for those who claim their voting rights have been violated; for desegregation or for the ' \
            'most extensive desegregation if alternative plans are at issue; for the rights of the racial minority ' \
            'or women (i.e., opposing the claim of reverse discrimination); for upholding the position of the person ' \
            'asserting the denial of their rights.',
        3: ' Consider liberal to be for assertion of broadest interpretation of ' \
            'First Amendment protection.',
        4: ' Consider liberal to be for interest of person asserting due process rights violated.',
        5: ' Consider liberal to be for interest of person asserting privacy rights violated.',
        6: ' Consider liberal in suits against ' \
            'management, for union, individual worker, or government in suit against management; in government ' \
            'enforcement of labor laws, for the federal government or the validity of federal regulations; in ' \
            'Executive branch vs union or workers, for executive branch; in worker vs union (non-civil rights), for ' \
            'union; in conflicts between rival union, for union which opposed by management and "not ascertained" if neither union ' \
            'supported by management or if unclear; in injured workers or consumers vs management, against management; ' \
            'in other labor issues, for economic underdog if no civil rights issue is present; for support of person ' \
            'claiming denial of civil rights.',
        7: ' Consider liberal to be for government tax claim; for person ' \
            'claiming patent or copyright infringement; for the plaintiff alleging the injury; for economic underdog if ' \
            'one party is clearly an underdog in comparison to the other, neither party is clearly an economic underdog; ' \
            'in cases pitting an individual against a business, the individual is presumed to be the economic underdog ' \
            'unless there is a clear indication in the opinion to the contrary; for debtor or bankrupt; for government or ' \
            'private party raising claim of violation of antitrust laws, or party opposing merger; for the economic underdog ' \
            'in private conflict over securities; for individual claiming a benefit from government; for government in disputes ' \
            'over government contracts and government seizure of property; for government regulation in government regulation of ' \
            'business; for greater protection of the environment or greater consumer protection (even if anti-government); for ' \
            'the injured party in admiralty - personal injury; for economic underdog in admiralty and miscellaneous economic cases.',
        9: ' Consider liberal to be  for assertion of federal power in federalism cases; "not ascertained" for conflict between states; ' \
            'for attorney; for the validity of challenged selective service regulation; or for the government interest in dispute ' \
            'with someone attempting to resist induction; for the authority of the challenged official in challenge to magistrates ' \
            'or referees; for defendant in Indian law - criminal; for the claim of the Indian or tribal rights in Indian law; for ' \
            'federal or state authority in Indian law vs state and federal authority; for interest of US or US firms when opposed by ' \
            'foreign firms or government; for US government if opposed to either US or foreign business in international law; for ' \
            'government regulation in immigration',
    }
}

issue_pre_tasks = {
    'geniss': {
        'name': 'songer_geniss',
        'instruction': f'{header}\n{header_issue} ' \
            'Consider the following categories: "criminal" (including appeals of conviction, petitions for post conviction ' \
            'relief, habeas corpus petitions, and other prisoner petitions which ' \
            'challenge the validity of the conviction or the sentence), ' \
            '"civil rights" (excluding First Amendment or due process; also excluding claims of denial of rights in criminal ' \
            'proceeding or claims by prisoners that challenge their conviction or their sentence (e.g., ' \
            'habeas corpus petitions are coded under the criminal category); ' \
            'does include civil suits instituted by both prisoners and callable ' \
            'non-prisoners alleging denial of rights by criminal justice officials), ' \
            '"First Amendment", "due process" (claims in civil cases by persons other than prisoners, ' \
            'does not include due process challenges to government economic regulation), ' \
            '"privacy", "labor relations", "economic activity and regulation", and "miscellaneous".',
        'question': 'What is the general issue in the case?',
        'answer_choices': {
            1: 'criminal',
            2: 'civil rights',
            3: 'First Amendment',
            4: 'due process',
            5: 'privacy',
            6: 'labor relations',
            7: 'economic activity and regulation',
            9: 'miscellaneous',
        },
    },
    'two_issues' : {
        'name': 'songer_two_issues',
        'instruction': f'{header}\n' \
            'Your task is to determine whether there are two issues in the case. ' \
            'By issue we mean the social and/or political context of the litigation in which more purely legal issues are argued. ' \
            'Put somewhat differently, this field identifies the nature of the conflict between the litigants. ' \
            'The focus here is on the subject matter of the controversy rather than its legal basis.',
        'question': 'Are there two issues in the case?',
        'get_decision': lambda case_: int(case_['casetyp2'] == case_['casetyp2']),
        'answer_choices': {
            0: 'no',
            1: 'yes',
        },
    },
    'treat': {
        'name': 'songer_treat',
        'instruction': f'{header}\n' \
            'Your task is to determine the disposition by the court of appeals of the decision of the court or agency below; i.e., how the decision below is "treated" by the appeals court. ' \
            'That is, the basic outcome of the case for the litigants, indicating whether the appellant or respondent "won" in the court of appeals.',
        'question': 'What is the disposition by the court of appeals of the decision of the court or agency below?',
        'answer_choices': {
            0: 'stay, petition, or motion granted',
            1: 'affirmed; or affirmed and petition denied',
            2: 'reversed (include reversed & vacated)',
            3: 'reversed and remanded (or just remanded)',
            4: 'vacated and remanded (also set aside & remanded; modified and remanded)',
            5: 'affirmed in part and reversed in part (or modified or affirmed and modified)',
            6: 'affirmed in part, reversed in part, and remanded; affirmed in part, vacated in part, and remanded',
            7: 'vacated',
            8: 'petition denied or appeal dismissed',
            9: 'certification to another court',
            10: 'not ascertained',
        },
    },
    'majvotes': {
        'name': 'songer_majvotes',
        'instruction': f'{header}\n' \
            'Your task is to determine the number of judges who voted in favor of the disposition favored by the majority. ' \
            'Judges who concurred in the outcome but wrote a separate concurring opinion are counted as part of the majority. ' \
            'For most cases this variable takes the value "2" or "3." However, for cases decided en banc the value may be as high as 15. ' \
            'Note: in the typical case, a list of the judges who heard the case is printed immediately before the opinion. ' \
            'If there is no indication that any of the judges dissented and no indication that one or more of the judges did not participate in the final decision, ' \
            'then all of the judges listed as participating in the decision are assumed to have cast votes with the majority. ' \
            'The number of majority votes recorded includes district judges or other judges sitting by designation who participated on the appeals court panel. ' \
            'If there is an indication that a judge heard argument in the case but did not participate in the final opinion (e.g., the judge died before the decision was reached), ' \
            'that judge is not counted in the number of majority votes.',
        'question': 'What is the number of judges who voted in favor of the disposition favored by the majority?',
        'answer_choices': {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15',
        },
    },
    'dissent': {
        'name': 'songer_dissent',
        'instruction': f'{header}\n' \
            'Your task is to determine the number of judges who dissented from the majority (either with or without opinion). ' \
            'Judges who dissented in part and concurred in part are counted as dissenting. ',
        'question': 'What is the number of judges who dissented from the majority?',
        'answer_choices': {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15',
        },
    },
    'concur': {
        'name': 'songer_concur',
        'instruction': f'{header}\n' \
            'Your task is to determine the number of judges who either wrote a concurring opinion, joined a concurring opinion, or who indicated that they concurred in the result but not in the opinion of the court.',
        'question': 'What is the number of judges who concurred in the result but not in the opinion of the court?',
        'answer_choices': {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15',
        },
    },
    'habeas': {
        'name': 'songer_habeas',
        'instruction': f'{header}\n' \
            'Your task is to determine whether the case was an appeal of a decision by the district court on a petition for habeas corpus. ' \
            'A state habeas corpus case is one in which a state inmate has petitioned the federal courts.',
        'question': 'Was the case an appeal of a decision by the district court on a petition for habeas corpus?',
        'answer_choices': {
            0: 'no',
            1: 'yes, state habeas corpus (criminal)',
            2: 'yes, federal habeas corpus (criminal)',
            3: 'yes, federal habeas corpus relating to deportation',
        },
    },
    'decuncon': {
        'name': 'songer_decuncon',
        'instruction': f'{header}\n' \
            'Your task is to determine whether the court declared any statute or administrative action unconstitutional. ' \
            'Only explicit statements in the opinion that some provision is unconstitutional should be used. ' \
            'Procedural violations of the constitution in the courts below are not counted as judicial review (e.g., if the trial court threw out evidence obtained in a search and seizure because of a 4th Amendment violation, the action would not count as judicial review).',
        'question': 'Did the court declare any statute or administrative action unconstitutional?',
        'answer_choices': {
            0: 'no declarations of unconstitutionality',
            1: 'act of Congress declared unconstitutional (facial invalidity)',
            2: 'interpretation/application of federal law invalid',
            3: 'federal administrative action or regulation unconstitutional on its face',
            4: 'interpretation/application of administrative regs unconstitutional',
            5: 'state constitution declared unconstitutional on its face',
            6: 'interpretation/application of state constitution unconstitutional',
            7: 'state law or regulation unconstitutional on its face',
            8: 'interpretation/application of state law/regulation unconstitutional',
            9: 'substate law or regulation unconstitutional on its face',
            10: 'interpretation/application of substate law/regulation unconstitutional',
        },
    },
    'typeiss': {
        'name': 'songer_typeiss',
        'instruction': f'{header}\n' \
            'Your task is to determine the general category of issues discussed in the opinion of the court. ' \
            'Choose among the following categories. Criminal and prisioner petitions- includes appeals of conviction, petitions for post conviction relief, habeas corpus petitions, and other prisoner petitions which challenge the validity of the conviction or the sentence or the validity of continued confinement. ' \
            'Civil - Government - these will include appeals from administrative agencies (e.g., OSHA,FDA), the decisions of administrative law judges, or the decisions of independent regulatory agencies (e.g., NLRB, FCC,SEC). The focus in administrative law is usually on procedural principles that apply to administrative agencies as they affect private interests, primarily through rulemaking and adjudication. Tort actions against the government, including petitions by prisoners which challenge the conditions of their confinement or which seek damages for torts committed by prion officials or by police fit in this category. In addition, this category will include suits over taxes and claims for benefits from government. ' \
            'Diversity of Citizenship - civil cases involving disputes between citizens of different states (remember that businesses have state citizenship). These cases will always involve the application of state or local law. If the case is centrally concerned with the application or interpretation of federal law then it is not a diversity case. ' \
            'Civil Disputes - Private - includes all civil cases that do not fit in any of the above categories. The opposing litigants will be individuals, businesses or groups.',
        'question': 'What is the general category of issues discussed in the opinion of the court?',
        'answer_choices': {
            1: 'criminal and prisoner petitions',
            2: 'civil - government',
            3: 'diversity of citizenship',
            4: 'civil - private',
            5: 'other, not applicable',
            0: 'not ascertained',
        },
    }
}

citing_cases = {
    'const1': {
        'name': 'songer_const1',
        'instruction': f'{header}\n' \
            'Your task is to identify the most frequently cited provision of the U.S. Constitution in the headnotes to this case. ' \
            'Answer "0" if no constitutional provisions are cited. ' \
            'If one or more are cited, code the article or amendment to the constitution which is mentioned in the greatest number of headnotes. ' \
            'In case of a tie, code the first mentioned provision of those that are tied. ' \
            'If it is one of the original articles of the constitution, code the number of the article preceeded by two zeros. ' \
            'If it is an amendment to the constitution, code the number of the amendment (zero filled to two places) preceeded by a "1". ' \
            'Examples: 001 = Article 1 of the original constitution, 101 = 1st Amendment, 114 = 14th Amendment.',
        'question': 'What is the most frequently cited provision of the U.S. Constitution in the headnotes to this case? ' \
            'If it is one of the original articles of the constitution, code the number of the article preceeded by two zeros. ' \
            'If it is an amendment to the constitution, code the number of the amendment (zero filled to two places) preceeded by a "1". ' \
            'Examples: 001 = Article 1 of the original constitution, 101 = 1st Amendment, 114 = 14th Amendment.',
        'type': 'int',
    },
    'const2': {
        'name': 'songer_const2',
        'instruction': f'{header}\n' \
            'Your task is to identify the second most frequently cited provision of the U.S. Constitution in the headnotes to this case. ' \
            'Answer "0" if fewer than two constitutional provisions are cited. ' \
            'If one or more are cited, code the article or amendment to the constitution which is mentioned in the second greatest number of headnotes. ' \
            'In case of a tie, code the second mentioned provision of those that are tied. ' \
            'If it is one of the original articles of the constitution, code the number of the article preceeded by two zeros. ' \
            'If it is an amendment to the constitution, code the number of the amendment (zero filled to two places) preceeded by a "1". ' \
            'Examples: 001 = Article 1 of the original constitution, 101 = 1st Amendment, 114 = 14th Amendment.',
        'question': 'What is the second most frequently cited provision of the U.S. Constitution in the headnotes to this case? ' \
            'If it is one of the original articles of the constitution, code the number of the article preceeded by two zeros. ' \
            'If it is an amendment to the constitution, code the number of the amendment (zero filled to two places) preceeded by a "1". ' \
            'Examples: 001 = Article 1 of the original constitution, 101 = 1st Amendment, 114 = 14th Amendment.',
        'type': 'int',
    },
    'usc1': {
        'name': 'songer_usc1',
        'instruction': f'{header}\n' \
            'Your task is to identify the most frequently cited title of the U.S. Code in the headnotes to this case. ' \
            'Answer "0" if no U.S. Code titles are cited. ' \
            'If one or more provisions are cited, code the number of the most frequently cited title.',
        'question': 'What is the most frequently cited title of the U.S. Code in the headnotes to this case? Answer with a number.',
        'type': 'int',
    },
    'usc2': {
        'name': 'songer_usc2',
        'instruction': f'{header}\n' \
            'The most frequently cited title of the U.S. Code in the headnotes to this case is {usc1}. ' \
            'Your task is to identify the second most frequently cited title of the U.S. ' \
            'Code in the headnotes to this case. ' \
            'Answer "0" if fewer than two U.S. Code titles are cited. ' \
            'To choose the second title, the following rule was used: If two or more titles of USC or USCA are cited, choose the second most frequently cited title, even if there are other sections of the title already coded which are mentioned more frequently. ' \
            'If the title already coded is the only title cited in the headnotes, choose the section of that title which is cited the second greatest number of times.',
        'question': 'The most frequently cited title of the U.S. Code in the headnotes to this case is {usc1}. ' \
            'What is the second most frequently cited title of this U.S. Code in the headnotes to this case? Answer with a number.',
        'ignore_targets': [0],
        'fill_in': ['usc1'],
        'type': 'int',
    },
    'usc1sect': {
        'name': 'songer_usc1sect',
        'instruction': f'{header}\n' \
            'Your task is to identify the number of the section from the title of the most frequently cited title of the U.S. Code in the headnotes to this case, ' \
            'that is, title {usc1}. ' \
            'In case of ties, code the first to be cited. The section number has up to four digits and follows "USC" or "USCA".',
        'question': 'What is the number of the section from the title of the most frequently cited title ' \
            'of the U.S. Code in the headnotes to this case, that is, title {usc1}? Answer with a number.',
        'ignore_targets': [0],
        'fill_in': ['usc1'],
        'type': 'int',
    },
    'usc2sect': {
        'name': 'songer_usc2sect',
        'instruction': f'{header}\n' \
            'Your task is to identify the number of the section from the title of the second most frequently cited title of the U.S. Code in the headnotes to this case, ' \
            'that is, title {usc2}. ' \
            'In case of ties, code the first to be cited. The section number has up to four digits and follows "USC" or "USCA".',
        'question': 'What is the number of the section from the title of the second most frequently cited title ' \
            'of the U.S. Code in the headnotes to this case, that is, title {usc2}? Answer with a number.',
        'ignore_targets': [0],
        'fill_in': ['usc2'],
        'type': 'int',
    },
    'civproc1': {
        'name': 'songer_civproc1',
        'instruction': f'{header}\n' \
            'Your task is to identify the most frequently cited federal rule of civil procedure in the headnotes to this case. ' \
            'Answer "0" if no federal rules of civil procedure are cited. ' \
            'For ties, code the first rule cited.',
        'question': 'What is the most frequently cited federal rule of civil procedure in the headnotes to this case? Answer with a number.',
        'type': 'int',
    },
    'civproc2': {
        'name': 'songer_civproc2',
        'instruction': f'{header}\n' \
            'Your task is to identify the second most frequently cited federal rule of civil procedure in the headnotes to this case. ' \
            'Answer "0" if less than two federal rules of civil procedure are cited. ' \
            'For ties, code the first rule cited.',
        'question': 'What is the second most frequently cited federal rule of civil procedure in the headnotes to this case? Answer with a number.',
        'type': 'int',
    },
    'crmproc1': {
        'name': 'songer_crmproc1',
        'instruction': f'{header}\n' \
            'Your task is to identify the most frequently cited federal rule of criminal procedure in the headnotes to this case. ' \
            'Answer "0" if no federal rules of criminal procedure are cited. ' \
            'For ties, code the first rule cited.',
        'question': 'What is the most frequently cited federal rule of criminal procedure in the headnotes to this case? Answer with a number.',
        'type': 'int',
    },
    'crmproc2': {
        'name': 'songer_crmproc2',
        'instruction': f'{header}\n' \
            'Your task is to identify the second most frequently cited federal rule of criminal procedure in the headnotes to this case. ' \
            'Answer "0" if less than two federal rules of criminal procedure are cited. ' \
            'For ties, code the first rule cited.',
        'question': 'What is the second most frequently cited federal rule of criminal procedure in the headnotes to this case? Answer with a number.',
        'type': 'int',
    },
}

instructions_issue = "Answer the question based on the directionality of the appeals court decision. " \
    "If the court discussed the issue in its opinion and answered the related question in the affirmative, answer \"Yes\". " \
    "If the issue was discussed and the opinion answered the question negatively, answer \"No\". " \
    "If the opinion considered the question but gave a mixed answer, supporting the respondent in part and supporting the appellant in part, answer \"Mixed answer\". " \
    "If the opinion does not discuss the issue, or notes that a particular issue was raised by one of the litigants but the court dismissed the issue as frivolous or trivial " \
    "or not worthy of discussion for some other reason, answer \"Issue not discussed\". " \
    "If the opinion considered the question but gave a \"mixed\" answer, supporting the respondent in part and supporting the appellant in part " \
    "(or if two issues treated separately by the court both fell within the area covered by one question and the court answered one question affirmatively and one negatively), " \
    "answer \"Mixed answer\". " \
    "If the opinion either did not consider or discuss the issue at all or if the opinion indicates that this issue was not worthy of consideration by the court of appeals even " \
    "though it was discussed by the lower court or was raised in one of the briefs, answer \"Issue not discussed\"."

instruction_criminal = f"If the court answered the question in the affirmative, but the error articulated by the court was judged to be harmless, answer \"Yes, but error was harmless\". "

header_threshold_trial = "You will be asked a question pertaining to some threshold issue at the trial court level. " \
    "These issues are only considered to be present if the court of appeals is reviewing whether or not the litigants should properly have been allowed to get a trial court decision on the merits. " \
    "That is, the issue is whether or not the issue crossed properly the threshhold to get on the district court agenda." \

header_threshold_appelate = "You will be asked a question pertaining to some threshold issue at the appeals court level. " \
    "That is, it is conceded that the trial court properly reached the merits, but the issue is whether, in spite of that concession, " \
    "the appellant has a right to an appeals court decision on the merits (e.g., the issue became moot after the trial). "

header_civil_law = "You will be asked a question pertaining to issues that may appear in any civil law cases including civil government, civil private, and diversity cases."
header_civil_gov = "You will be asked a question pertaining to issues that may appear in civil law issues involving government actors."

answer_choices_all = {
    1: "No",
    2: "Yes",
    9: "Mixed answer",
    0: "Issue not discussed"
}

answer_choices_criminal = {
    1: "No",
    2: "Yes",
    3: "Yes, but error was harmless",
    9: "Mixed answer",
    0: "Issue not discussed"
}

answer_choices_constit = {
    0: "Issue not discussed",
    1: "The issue was discussed in the opinion and the resolution of the issue by the court favored the respondent",
    2: "The issue was discussed in the opinion and the resolution of the issue by the court favored the appellant",
    9: "The resolution of the issue had mixed results for the appellant and respondent"
}


issue_tasks = [
    ('constit', {
        'name': 'songer_constit',
        'instruction': f'{header} Your task is to determine whether there was an issue discussed in the opinion of the court about the ' \
            'constitutionality of a law or administrative action, and if so, whether the resolution of the issue by the court favored the appellant.',
        'question': 'Did the court\'s conclusion about the constitutionality of a law or administrative action favor the appellant?',
        'answer_choices': answer_choices_constit,
    }),
    ('fedlaw', {
        'name': 'songer_fedlaw',
        'instruction': f'{header} Your task is to determine whether there was an issue discussed in the opinion of the court about the ' \
            'interpretation of federal statute, and if so, whether the resolution of the issue by the court favored the appellant.',
        'question': 'Did the interpretation of federal statute by the court favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    ('procedur', {
        'name': 'songer_procedur',
        'instruction': f'{header} Your task is to determine whether there was an issue discussed in the opinion of the court about the ' \
            'interpretation of federal rule of procedures, judicial doctrine, or case law, and if so, whether the resolution of the issue by the court favored the appellant.',
        'question': 'Did the interpretation of federal rule of procedures, judicial doctrine, or case law by the court favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    ('juris', {
        'name': 'songer_jurisdiction',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court determine that it had jurisdiction to hear this case?" {instructions_issue}' \
                        'If the opinion discusses challenges to the jurisdiction of the court to hear several different issues and the court ruled that it had jurisdiction ' \
                        'to hear some of the issues but did not have jurisdiction to hear other issues, answer "Mixed answer". ',
        'question': 'Did the court determine that it had jurisdiction to hear this case?',
        'answer_choices': answer_choices_all,
    }),
    ('statecl', {
        'name': 'songer_stateclaim',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court dismiss the case because of the failure of the plaintiff to state a claim upon which relief could be granted?" {instructions_issue}' \
                        'The issue hereby considered also pertains to cases where the court concluded that there was no proper cause of action.',
        'question': 'Did the court dismiss the case because of the failure of the plaintiff to state a claim upon which relief could be granted?',
        'answer_choices': answer_choices_all,
    }),
    ('standing', {
        'name': 'songer_standing',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court determine that the parties had standing?" {instructions_issue}',
        'question': 'Did the court determine that the parties had standing?',
        'answer_choices': answer_choices_all,
    }),
    ('mootness', {
        'name': 'songer_mootness',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court conclude that an issue was moot?" {instructions_issue}',
        'question': 'Did the court conclude that an issue was moot?',
        'answer_choices': answer_choices_all,
    }),
    # EXHAUST - Did the court determine that it would not hear the appeal for one of the following reasons : a)administrative remedies had not been exhausted; or b) the issue was not ripe for judicial action ?
    ('exhaust', {
        'name': 'songer_exhaust',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court determine that it would not hear the appeal for one of the following reasons: a) administrative remedies had not been exhausted; or b) the issue was not ripe for judicial action?" {instructions_issue}',
        'question': 'Did the court determine that it would not hear the appeal for one of the following reasons: a) administrative remedies had not been exhausted; or b) the issue was not ripe for judicial action?',
        'answer_choices': answer_choices_all,
    }),
    # TIMELY - Did the court conclude that it could not reach the merits of the case because the litigants had not complied with some rule relating to timeliness, a filing fee, or because a statute of limitations had expired ?
    ('timely', {
        'name': 'songer_timely',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court conclude that it could not reach the merits of the case because the litigants had not complied with some rule relating to timeliness, a filing fee, or because a statute of limitations had expired?" {instructions_issue}',
        'question': 'Did the court conclude that it could not reach the merits of the case because the litigants had not complied with some rule relating to timeliness, a filing fee, or because a statute of limitations had expired?',
        'answer_choices': answer_choices_all,
    }),
    # IMMUNITY - Did the court refuse to reach the merits of the appeal because it concluded that the defendant had immunity (e.g., the governmental immunity doctrine) ?
    ('immunity', {
        'name': 'songer_immunity',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court refuse to reach the merits of the appeal because it concluded that the defendant had immunity (e.g., the governmental immunity doctrine)?" {instructions_issue}',
        'question': 'Did the court refuse to reach the merits of the appeal because it concluded that the defendant had immunity?',
        'answer_choices': answer_choices_all,
    }),
    # FRIVOL - Did the court conclude that either the original case was frivolous or raised only trivial issues and therefore was not suitable for actions on the merits ?
    ('frivol', {
        'name': 'songer_frivol',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court conclude that either the original case was frivolous or raised only trivial issues and therefore was not suitable for actions on the merits?" {instructions_issue}',
        'question': 'Did the court conclude that either the original case was frivolous or raised only trivial issues and therefore was not suitable for actions on the merits?',
        'answer_choices': answer_choices_all,
    }),
    # POLQUEST - Did the court refuse to rule on the merits of the case because it was considered to be a nonjusticiable "political question" ?
    ('polquest', {
        'name': 'songer_polquest',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court refuse to rule on the merits of the case because it was considered to be a nonjusticiable "political question"?" {instructions_issue}',
        'question': 'Did the court refuse to rule on the merits of the case because it was considered to be a nonjusticiable "political question"?',
        'answer_choices': answer_choices_all,
    }),
    # OTHTHRES - Did the court refuse to rule on the merits of the appeal because of a threshhold issue other than lack of jurisdiction, standing, mootness, failure to state a claim, exhaustion, timeliness, immunity, frivolousness, or nonjusticiable political question ?
    ('oththres', {
        'name': 'songer_oththres',
        'instruction': f'{header} {header_threshold_trial} The issue is: "Did the court refuse to rule on the merits of the appeal because of a threshhold issue other than lack of jurisdiction, standing, mootness, failure to state a claim, exhaustion, timeliness, immunity, frivolousness, or nonjusticiable political question?" {instructions_issue}',
        'question': 'Did the court refuse to rule on the merits of the appeal because of a threshhold issue other than lack of jurisdiction, standing, mootness, failure to state a claim, exhaustion, timeliness, immunity, frivolousness, or nonjusticiable political question?',
        'answer_choices': answer_choices_all,
    }),
    # LATE - Did the court refuse to decide the appeal because the appellant failed to comply with some rule relating to timeliness of the appeal (e.g., failed to pay the filing fee on time or missed the deadline to file the appeal)?
    ('late', {
        'name': 'songer_late',
        'instruction': f'{header} {header_threshold_appelate} The issue is: "Did the court refuse to decide the appeal because the appellant failed to comply with some rule relating to timeliness of the appeal (e.g., failed to pay the filing fee on time or missed the deadline to file the appeal)?" {instructions_issue}',
        'question': 'Did the court refuse to decide the appeal because the appellant failed to comply with some rule relating to timeliness of the appeal?',
        'answer_choices': answer_choices_all,
    }),
    # FRIVAPP - Did the court conclude that it could not reach the merits of the case because the motion or appeal was frivolous or raised only trivial issues and was therefore not suitable for appellate review?
    ('frivapp', {
        'name': 'songer_frivapp',
        'instruction': f'{header} {header_threshold_appelate} The issue is: "Did the court conclude that it could not reach the merits of the case because the motion or appeal was frivolous or raised only trivial issues and was therefore not suitable for appellate review?" {instructions_issue}',
        'question': 'Did the court conclude that it could not reach the merits of the case because the motion or appeal was frivolous or raised only trivial issues and was therefore not suitable for appellate review?',
        'answer_choices': answer_choices_all,
    }),
    ('othappth', {
        'name': 'songer_othappth',
        'instruction': f'{header} {header_threshold_appelate} The issue is: "Did the court refuse to rule on the merits of the appeal because of some threshhold issue other than timeliness or frivolousness that was relevant on appeal but not at the original trial? (e.g., the case became moot after the original trial)" {instructions_issue}',
        'question': 'Did the court refuse to rule on the merits of the appeal because of some threshhold issue other than timeliness or frivolousness that was relevant on appeal but not at the original trial?',
        'answer_choices': answer_choices_all,
    }),
    # criminal cases
    # PREJUD - Was there prejudicial conduct by prosecution ? (including prosecutor refusing to produce evidence which would aid defendant)
    ('prejud', {
        'name': 'songer_prejud',
        'instruction': f'{header} The issue is: "Was there prejudicial conduct by prosecution? (including prosecutor refusing to produce evidence which would aid defendant)" {instructions_issue} {instruction_criminal}',
        'question': 'Was there prejudicial conduct by prosecution?',
        'answer_choices': answer_choices_criminal,
    }),
    # INSANE - Did the court below err in not permitting an insanity defense? (or did the court err in its conclusion about whether the defendan was mentally competent to stand trial)
    ('insane', {
        'name': 'songer_insane',
        'instruction': f'{header} The issue is: "Did the court below err in not permitting an insanity defense? (or did the court err in its conclusion about whether the defendan was mentally competent to stand trial)" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court below err in not permitting an insanity defense?',
        'answer_choices': answer_choices_criminal,
    }),
    # IMPROPER - Did the court conclude that there was improper influence on the jury? For example, jury tampering or failure to shield jury from prejudicial media accounts. Exclude prejudicial conduct by the prosecutor.
    ('improper', {
        'name': 'songer_improper',
        'instruction': f'{header} The issue is: "Did the court conclude that there was improper influence on the jury? For example, include jury tampering or failure to shield jury from prejudicial media accounts. Exclude prejudicial conduct by the prosecutor." {instructions_issue} {instruction_criminal}',
        'question': 'Did the court conclude that there was improper influence on the jury? For example, include jury tampering or failure to shield jury from prejudicial media accounts. Exclude prejudicial conduct by the prosecutor.',
        'answer_choices': answer_choices_criminal,
    }),
    # JURYINST - Did the court conclude that the jury instructions were improper?
    ('juryinst', {
        'name': 'songer_juryinst',
        'instruction': f'{header} The issue is: "Did the court conclude that the jury instructions were improper?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court conclude that the jury instructions were improper?',
        'answer_choices': answer_choices_criminal,
    }),
    # OTHJURY - Did the court conclude that the jury composition or selection was invalid or that the jury was biased or tampered with?
    ('othjury', {
        'name': 'songer_othjury',
        'instruction': f'{header} The issue is: "Did the court conclude that the jury composition or selection was invalid or that the jury was biased or tampered with?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court conclude that the jury composition or selection was invalid or that the jury was biased or tampered with?',
        'answer_choices': answer_choices_criminal,
    }),
    # DEATHPEN - Did the court conclude that the death penalty was improperly imposed? Consider only the validity of the sentence, rather than whether or not the conviction was proper.
    ('deathpen', {
        'name': 'songer_deathpen',
        'instruction': f'{header} The issue is: "Did the court conclude that the death penalty was improperly imposed? Consider only the validity of the sentence, rather than whether or not the conviction was proper." {instructions_issue} {instruction_criminal}',
        'question': 'Did the court conclude that the death penalty was improperly imposed? Consider only the validity of the sentence, rather than whether or not the conviction was proper.',
        'answer_choices': answer_choices_criminal,
    }),
    # SENTENCE - Did the court conclude that some penalty, excluding the death penalty, was improperly imposed?
    ('sentence', {
        'name': 'songer_sentence',
        'instruction': f'{header} The issue is: "Did the court conclude that some penalty, excluding the death penalty, was improperly imposed?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court conclude that some penalty, excluding the death penalty, was improperly imposed?',
        'answer_choices': answer_choices_criminal,
    }),
    # INDICT - Did the court rule that the indictment was defective ?
    ('indict', {
        'name': 'songer_indict',
        'instruction': f'{header} The issue is: "Did the court rule that the indictment was defective?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that the indictment was defective?',
        'answer_choices': answer_choices_criminal,
    }),
    # CONFESS - Did the court conclude that a confession or an incriminating statement was improperly admitted? Consider only incriminating statements made by the defendant.
    ('confess', {
        'name': 'songer_confess',
        'instruction': f'{header} The issue is: "Did the court conclude that a confession or an incriminating statement was improperly admitted? Consider only incriminating statements made by the defendant." {instructions_issue} {instruction_criminal}',
        'question': 'Did the court conclude that a confession or an incriminating statement was improperly admitted? Consider only incriminating statements made by the defendant.',
        'answer_choices': answer_choices_criminal,
    }),
    # SEARCH - Did the court below improperly rule for the prosecution on an issue related to an alleged illegal search and seizure?
    ('search', {
        'name': 'songer_search',
        'instruction': f'{header} The issue is: "Did the court below improperly rule for the prosecution on an issue related to an alleged illegal search and seizure?" {instructions_issue} {instruction_criminal}' \
            "If a civil suit brought by a prisoner or a criminal defendant in another action that alleges a tort based on an illegal search and seizure, also consider the issue to be present in the case.",
        'question': 'Did the court below improperly rule for the prosecution on an issue related to an alleged illegal search and seizure?',
        'answer_choices': answer_choices_criminal,
    }),
    # OTHADMIS - Did the court rule that some evidence, other than a confession made by the defendant or illegal search and seizure, was inadmissibile (or did ruling on appropriateness of evidentary hearing benefit the defendant)?
    ('othadmis', {
        'name': 'songer_othadmis',
        'instruction': f'{header} The issue is: "Did the court rule that some evidence, other than a confession made by the defendant or illegal search and seizure, was inadmissibile, (or did ruling on appropriateness of evidentary hearing benefit the defendant)?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that some evidence, other than a confession made by the defendant or illegal search and seizure, was inadmissibile (or did ruling on appropriateness of evidentary hearing benefit the defendant)?',
        'answer_choices': answer_choices_criminal,
    }),
    # PLEA - Did the court rule for the defendant on an issue related to plea bargaining? Plea bargain includes all challenges to plea.
    ('plea', {
        'name': 'songer_plea',
        'instruction': f'{header} The issue is: "Did the court rule for the defendant on an issue related to plea bargaining? Plea bargain includes all challenges to plea." {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule for the defendant on an issue related to plea bargaining? Plea bargain includes all challenges to plea.',
        'answer_choices': answer_choices_criminal,
    }),
    # COUNSEL - Did the court rule that the defendant had inadequate counsel?
    ('counsel', {
        'name': 'songer_counsel',
        'instruction': f'{header} The issue is: "Did the court rule that the defendant had inadequate counsel?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that the defendant had inadequate counsel?',
        'answer_choices': answer_choices_criminal,
    }),
    # RTCOUNS - Did the court rule that the defendant's right to counsel was violated (for some reason other than inadequate counsel)?
    ('rtcouns', {
        'name': 'songer_rtcouns',
        'instruction': f'{header} The issue is: "Did the court rule that the defendant\'s right to counsel was violated (for some reason other than inadequate counsel)?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that the defendant\'s right to counsel was violated (for some reason other than inadequate counsel)?',
        'answer_choices': answer_choices_criminal,
    }),
    # SUFFIC - Did the court rule that there was insufficient evidence for conviction ?
    ('suffic', {
        'name': 'songer_suffic',
        'instruction': f'{header} The issue is: "Did the court rule that there was insufficient evidence for conviction?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that there was insufficient evidence for conviction?',
        'answer_choices': answer_choices_criminal,
    }),
    # INDIGENT - Did the court rule that the defendant's rights as an indigent were violated?
    ('indigent', {
        'name': 'songer_indigent',
        'instruction': f'{header} The issue is: "Did the court rule that the defendant\'s rights as an indigent were violated?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that the defendant\'s rights as an indigent were violated?',
        'answer_choices': answer_choices_criminal,
    }),
    # ENTRAP - Did the court rule that the defendant was the victim of illegal entrapment?
    ('entrap', {
        'name': 'songer_entrap',
        'instruction': f'{header} The issue is: "Did the court rule that the defendant was the victim of illegal entrapment?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule that the defendant was the victim of illegal entrapment?',
        'answer_choices': answer_choices_criminal,
    }),
    # PROCDIS - Did the court uphold the dismissal by district court on procedural grounds ?
    ('procdis', {
        'name': 'songer_procdis',
        'instruction': f'{header} The issue is: "Did the court uphold the dismissal by district court on procedural grounds?" {instructions_issue} {instruction_criminal}',
        'question': 'Did the court uphold the dismissal by district court on procedural grounds?',
        'answer_choices': answer_choices_criminal,
    }),
    # OTHCRIM - Did the court rule for the defendant on grounds other than procedural grounds? For example, right to speedy trial, double jeopardy, confrontation, retroactivity, self defense. This includes the question of whether the defendant waived the right to raise some claim.
    ('othcrim', {
        'name': 'songer_othcrim',
        'instruction': f'{header} The issue is: "Did the court rule for the defendant on grounds other than procedural grounds? For example, right to speedy trial, double jeopardy, confrontation, retroactivity, self defense." This includes the question of whether the defendant waived the right to raise some claim. {instructions_issue} {instruction_criminal}',
        'question': 'Did the court rule for the defendant on grounds other than procedural grounds? For example, right to speedy trial, double jeopardy, confrontation, retroactivity, self defense. This includes the question of whether the defendant waived the right to raise some claim.',
        'answer_choices': answer_choices_criminal,
    }),
    # civil law cases
    # DUEPROC - Did the interpretation of the requirements of due process by the court favor the appellant ?
    ('dueproc', {
        'name': 'songer_dueproc',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the interpretation of the requirements of due process by the court favor the appellant?" {instructions_issue}',
        'question': 'Did the interpretation of the requirements of due process by the court favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # EXECORD - Did the interpretation of executive order or administrative regulation by the court favor the appellant? This does include whether or not an executive order was lawful.
    ('execord', {
        'name': 'songer_execord',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the interpretation of executive order or administrative regulation by the court favor the appellant?" This does include whether or not an executive order was lawful. {instructions_issue}',
        'question': 'Did the interpretation of executive order or administrative regulation by the court favor the appellant? This does include whether or not an executive order was lawful.',
        'answer_choices': answer_choices_all,
    }),
    # STPOLICY - Did the interpretation of state or local law, executive order, administrative regulation, doctrine, or rule of procedure by the court favor the appellant ?
    ('stpolicy', {
        'name': 'songer_stpolicy',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the interpretation of state or local law, executive order, administrative regulation, doctrine, or rule of procedure by the court favor the appellant?" {instructions_issue}',
        'question': 'Did the interpretation of state or local law, executive order, administrative regulation, doctrine, or rule of procedure by the court favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # WEIGHTEV - Did the factual interpretation by the court or its conclusions (e.g., regarding the weight of evidence or the sufficiency of evidence) favor the appellant ? This includes discussions of whether the litigant met the burden of proof.
    ('weightev', {
        'name': 'songer_weightev',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the factual interpretation by the court or its conclusions (e.g., regarding the weight of evidence or the sufficiency of evidence) favor the appellant?" This includes discussions of whether the litigant met the burden of proof. {instructions_issue}',
        'question': 'Did the factual interpretation by the court or its conclusions (e.g., regarding the weight of evidence or the sufficiency of evidence) favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # PRETRIAL - Did the court's rulings on pre-trial procedure favor the appellant? This includes whether or not there is a right to jury trial, whether the case should be certified as a class action, or whether a prospective party has a right to intervene in the case, but does not include rulings on motions for summary judgment.
    ('pretrial', {
        'name': 'songer_pretrial',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s rulings on pre-trial procedure favor the appellant?" This includes whether or not there is a right to jury trial, whether the case should be certified as a class action, or whether a prospective party has a right to intervene in the case, but does not include rulings on motions for summary judgment. {instructions_issue}',
        'question': 'Did the court\'s rulings on pre-trial procedure favor the appellant? This includes whether or not there is a right to jury trial, whether the case should be certified as a class action, or whether a prospective party has a right to intervene in the case, but does not include rulings on motions for summary judgment.',
        'answer_choices': answer_choices_all,
    }),
    # TRIALPRO - Did the court's ruling on procedure at trial favor the appellant? This includes jury instructions and motions for directed verdicts made during trial.
    ('trialpro', {
        'name': 'songer_trialpro',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on procedure at trial favor the appellant?" This includes jury instructions and motions for directed verdicts made during trial. {instructions_issue}',
        'question': 'Did the court\'s ruling on procedure at trial favor the appellant? This includes jury instructions and motions for directed verdicts made during trial.',
        'answer_choices': answer_choices_all,
    }),
    # POST_TRL -  Did the court's ruling on some post-trial procedure or motion (e.g., allocating court costs or post award relief) favor the appellant? This doe not include attorneys' fees, but does include motions to set aside a jury verdict.
    ('post_trl', {
        'name': 'songer_post_trl',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on some post-trial procedure or motion (e.g., allocating court costs or post award relief) favor the appellant?" This doe not include attorneys\' fees, but does include motions to set aside a jury verdict. {instructions_issue}',
        'question': 'Did the court\'s ruling on some post-trial procedure or motion (e.g., allocating court costs or post award relief) favor the appellant? This doe not include attorneys\' fees, but does include motions to set aside a jury verdict.',
        'answer_choices': answer_choices_all,
    }),
    # ATTYFEE - Did the court's ruling on attorneys' fees favor the appellant?
    ('attyfee', {
        'name': 'songer_attyfee',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on attorneys\' fees favor the appellant?" {instructions_issue}',
        'question': 'Did the court\'s ruling on attorneys\' fees favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # JUDGDISC - Did the court's ruling on the abuse of discretion by the trial judge favor the appellant? This includes the issue of whether the judge actually had the authority for the action taken, but does not include questions of discretion of administrative law judges.
    ('judgdisc', {
        'name': 'songer_judgdisc',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on the abuse of discretion by the trial judge favor the appellant?" This includes the issue of whether the judge actually had the authority for the action taken, but does not include questions of discretion of administrative law judges. {instructions_issue}',
        'question': 'Did the court\'s ruling on the abuse of discretion by the trial judge favor the appellant? This includes the issue of whether the judge actually had the authority for the action taken, but does not include questions of discretion of administrative law judges.',
        'answer_choices': answer_choices_all,
    }),
    # ALTDISP - Did the court's ruling on an issue arising out of an alternative dispute resolution process (ADR, settlement conference, role of mediator or arbitrator, etc.) favor the appellant?
    ('altdisp', {
        'name': 'songer_altdisp',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on an issue arising out of an alternative dispute resolution process (ADR, settlement conference, role of mediator or arbitrator, etc.) favor the appellant?" {instructions_issue}',
        'question': 'Did the court\'s ruling on an issue arising out of an alternative dispute resolution process (ADR, settlement conference, role of mediator or arbitrator, etc.) favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # INJUNCT - Did the court's ruling on the validity of an injunction or the denial of an injunction or a stay of injunction favor the appellant?
    ('injunct', {
        'name': 'songer_injunct',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on the validity of an injunction or the denial of an injunction or a stay of injunction favor the appellant?" {instructions_issue}',
        'question': 'Did the court\'s ruling on the validity of an injunction or the denial of an injunction or a stay of injunction favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # SUMMARY - Did the court's ruling on the appropriateness of summary judgment or the denial of summary judgment favor the appellant?
    ('summary', {
        'name': 'songer_summary',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s ruling on the appropriateness of summary judgment or the denial of summary judgment favor the appellant?" {instructions_issue}',
        'question': 'Did the court\'s ruling on the appropriateness of summary judgment or the denial of summary judgment favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # FEDVST - Did the court rule that federal law should take precedence over state or local laws in a case involving the conflict of laws (i.e, which laws or rules apply)?
    ('fedvst', {
        'name': 'songer_fedvst',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court rule that federal law should take precedence over state or local laws in a case involving the conflict of laws (i.e, which laws or rules apply)?" {instructions_issue}',
        'question': 'Did the court rule that federal law should take precedence over state or local laws in a case involving the conflict of laws (i.e, which laws or rules apply)?',
        'answer_choices': answer_choices_all,
    }),
    # FOREIGN - Did the court rule that domestic law (federal, state or local) should take precedence over foreign law in a case involving the conflict of laws (i.e., which laws or rules apply- foreign country vs federal, state, or local)?
    ('foreign', {
        'name': 'songer_foreign',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court rule that domestic law (federal, state or local) should take precedence over foreign law in a case involving the conflict of laws (i.e., which laws or rules apply- foreign country vs federal, state, or local)?" {instructions_issue}',
        'question': 'Did the court rule that domestic law (federal, state or local) should take precedence over foreign law in a case involving the conflict of laws (i.e., which laws or rules apply- foreign country vs federal, state, or local)?',
        'answer_choices': answer_choices_all,
    }),
    # INT_LAW - Did the court rule in favor of the appellant on an issue related to the interpretation of a treaty or international law?
    ('int_law', {
        'name': 'songer_int_law',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court rule in favor of the appellant on an issue related to the interpretation of a treaty or international law?" {instructions_issue}',
        'question': 'Did the court rule in favor of the appellant on an issue related to the interpretation of a treaty or international law?',
        'answer_choices': answer_choices_all,
    }),
    # ST_V_ST - Did the court rule in favor of the appellant on the issue of a conflict of laws ( which laws or rules apply ) other than federal v state or foreign v domestic (e.g., one state vs second state) ?
    ('st_v_st', {
        'name': 'songer_st_v_st',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court rule in favor of the appellant on the issue of a conflict of laws ( which laws or rules apply ) other than federal v state or foreign v domestic (e.g., one state vs second state)?" {instructions_issue}',
        'question': 'Did the court rule in favor of the appellant on the issue of a conflict of laws ( which laws or rules apply ) other than federal v state or foreign v domestic (e.g., one state vs second state)?',
        'answer_choices': answer_choices_all,
    }),
    # DISCOVER - Did the court's interpretation of rules relating to discovery or other issues related to obtaining evidence favor the appellant?
    ('discover', {
        'name': 'songer_discover',
        'instruction': f'{header} {header_civil_law} The issue is: "Did the court\'s interpretation of rules relating to discovery or other issues related to obtaining evidence favor the appellant?" {instructions_issue}',
        'question': 'Did the court\'s interpretation of rules relating to discovery or other issues related to obtaining evidence favor the appellant?',
        'answer_choices': answer_choices_all,
    }),
    # SUBEVID - Did the court's interpretation of the substantial evidence rule support the government? For example, "such evidence as a reasonable mind might accept as adequate to support a conclusion" or "more than a mere scintilla". This issue is present only when the court indicates that it is using this doctrine, rather than when the court is merely discussing the evidence to determine whether the evidence supports the position of the appellant or respondent.
    ('subevid', {
        'name': 'songer_subevid',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court\'s interpretation of the substantial evidence rule support the government? For example, "such evidence as a reasonable mind might accept as adequate to support a conclusion" or "more than a mere scintilla". This issue is present only when the court indicates that it is using this doctrine, rather than when the court is merely discussing the evidence to determine whether the evidence supports the position of the appellant or respondent." {instructions_issue}',
        'question': 'Did the court\'s interpretation of the substantial evidence rule support the government? For example, "such evidence as a reasonable mind might accept as adequate to support a conclusion" or "more than a mere scintilla". This issue is present only when the court indicates that it is using this doctrine, rather than when the court is merely discussing the evidence to determine whether the evidence supports the position of the appellant or respondent.',
        'answer_choices': answer_choices_all,
    }),
    # DENOVO - Did the court's use of the standard of review, "de novo on facts" support the government? The courts generally recognize that de novo review is impractical for the bulk of agency decisions so the substantial evidence standard helps provide a middle course. Consider the de novo review of administrative action, not de novo review of trial court by appeals court.
    ('denovo', {
        'name': 'songer_denovo',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court\'s use of the standard of review, "de novo on facts" support the government?" The courts generally recognize that de novo review is impractical for the bulk of agency decisions so the substantial evidence standard helps provide a middle course. Consider the de novo review of administrative action, not de novo review of trial court by appeals court. {instructions_issue}',
        'question': 'Did the court\'s use of the standard of review, "de novo on facts" support the government? The courts generally recognize that de novo review is impractical for the bulk of agency decisions so the substantial evidence standard helps provide a middle course. Consider the de novo review of administrative action, not de novo review of trial court by appeals court.',
        'answer_choices': answer_choices_all,
    }),
    # ERRON - Did the court's use of the clearly erroneous standard support the government? That is, a somewhat narrower standard than substantial evidence, or ignoring usual agency standards.
    ('erron', {
        'name': 'songer_erron',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court\'s use of the clearly erroneous standard support the government?" That is, a somewhat narrower standard than substantial evidence, or ignoring usual agency standards. {instructions_issue}',
        'question': 'Did the court\'s use of the clearly erroneous standard support the government? That is, a somewhat narrower standard than substantial evidence, or ignoring usual agency standards.',
        'answer_choices': answer_choices_all,
    }),
    # CAPRIC - Did the courts's use or interpretation of the arbitrary and capricious standard support the government ? Note that APA allows courts to overturn agency actions deemed to be arbitrary or capricious, an abuse of discretion, or otherwise not in accordance with law. Overton Park emphasized this is a narrow standard, and one must prove that agency's action is without a rational basis. This also includes the "substantial justification" doctrine.
    ('capric', {
        'name': 'songer_capric',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the courts\'s use or interpretation of the arbitrary and capricious standard support the government? Note that APA allows courts to overturn agency actions deemed to be arbitrary or capricious, an abuse of discretion, or otherwise not in accordance with law. Overton Park emphasized this is a narrow standard, and one must prove that agency\'s action is without a rational basis. This also includes the "substantial justification" doctrine. {instructions_issue}',
        'question': 'Did the courts\'s use or interpretation of the arbitrary and capricious standard support the government? Note that APA allows courts to overturn agency actions deemed to be arbitrary or capricious, an abuse of discretion, or otherwise not in accordance with law. Overton Park emphasized this is a narrow standard, and one must prove that agency\'s action is without a rational basis. This also includes the "substantial justification" doctrine.',
        'answer_choices': answer_choices_all,
    }),
    # ABUSEDIS - Did the court conclude that it should defer to agency discretion? For example, if the action was committed to agency discretion.
    ('abusedis', {
        'name': 'songer_abusedis',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court conclude that it should defer to agency discretion? For example, if the action was committed to agency discretion. {instructions_issue}',
        'question': 'Did the court conclude that it should defer to agency discretion? For example, if the action was committed to agency discretion.',
        'answer_choices': answer_choices_all,
    }),
    # JUDREV - Did the court conclude the decision was subject to judicial review? While questions of fact are subject to limited review, questions of law are subject to full review. The problem becomes determining which are clear questions of law or fact as they are often "mixed".
    ('judrev', {
        'name': 'songer_judrev',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court conclude the decision was subject to judicial review?" While questions of fact are subject to limited review, questions of law are subject to full review. The problem becomes determining which are clear questions of law or fact as they are often "mixed". {instructions_issue}',
        'question': 'Did the court conclude the decision was subject to judicial review? While questions of fact are subject to limited review, questions of law are subject to full review. The problem becomes determining which are clear questions of law or fact as they are often "mixed".',
        'answer_choices': answer_choices_all,
    }),
    # GENSTAND - Did the agency articulate the appropriate general standard? This question includes whether the agency interpreted the statute "correctly". The courts often refer here to the rational basis test, plain meaning, reasonable construction of the statute, congressional intent, etc. This issue also includes question of which law applies or whether amended law vs law before amendment applies.
    ('genstand', {
        'name': 'songer_genstand',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the agency articulate the appropriate general standard?" This question includes whether the agency interpreted the statute "correctly". The courts often refer here to the rational basis test, plain meaning, reasonable construction of the statute, congressional intent, etc. This issue also includes question of which law applies or whether amended law vs law before amendment applies. {instructions_issue}',
        'question': 'Did the agency articulate the appropriate general standard? This question includes whether the agency interpreted the statute "correctly". The courts often refer here to the rational basis test, plain meaning, reasonable construction of the statute, congressional intent, etc. This issue also includes question of which law applies or whether amended law vs law before amendment applies.',
        'answer_choices': answer_choices_all,
    }),
    # NOTICE - Decisions that affect life, liberty, or property must be preceded by adequate notice and an opportunity for a fair hearing. Did the agency give proper notice?
    ('notice', {
        'name': 'songer_notice',
        'instruction': f'{header} {header_civil_gov} The issue is: "Decisions that affect life, liberty, or property must be preceded by adequate notice and an opportunity for a fair hearing. Did the agency give proper notice? {instructions_issue}',
        'question': 'Decisions that affect life, liberty, or property must be preceded by adequate notice and an opportunity for a fair hearing. Did the agency give proper notice?',
        'answer_choices': answer_choices_all,
    }),
    # ALJ - Did the court support the decision of an administrative law judge ?
    ('alj', {
        'name': 'songer_alj',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court support the decision of an administrative law judge? {instructions_issue}',
        'question': 'Did the court support the decision of an administrative law judge?',
        'answer_choices': answer_choices_all,
    }),
    # AGEN_ACQ - Did the court rule for the government in an issue related to agency acquisition of information (e.g. physical inspections, searches, subpoenas, records, etc)?
    ('agen_acq', {
        'name': 'songer_agen_acq',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court rule for the government in an issue related to agency acquisition of information (e.g. physical inspections, searches, subpoenas, records, etc)? {instructions_issue}',
        'question': 'Did the court rule for the government in an issue related to agency acquisition of information (e.g. physical inspections, searches, subpoenas, records, etc)?',
        'answer_choices': answer_choices_all,
    }),
    # FREEINFO - Did the court rule in favor of the government when the administrative action in question related to the agency's providing information to those who request it? For example, Freedom of Information, issues of governmental confidentiality, or "government in the sunshine".
    ('freeinfo', {
        'name': 'songer_freeinfo',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the court rule in favor of the government when the administrative action in question related to the agency\'s providing information to those who request it? For example, Freedom of Information, issues of governmental confidentiality, or "government in the sunshine". {instructions_issue}',
        'question': 'Did the court rule in favor of the government when the administrative action in question related to the agency\'s providing information to those who request it? For example, Freedom of Information, issues of governmental confidentiality, or "government in the sunshine".',
        'answer_choices': answer_choices_all,
    }),
    # COMMENT - Did the agency give proper opportunity to comment?
    ('comment', {
        'name': 'songer_comment',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the agency give proper opportunity to comment? {instructions_issue}',
        'question': 'Did the agency give proper opportunity to comment?',
        'answer_choices': answer_choices_all,
    }),
    # RECORD - Did the agency fail to develop an adequate record ? For example, if the court was unable to determine what doctrine was used for the decision or unable to determine the basis of the decision.
    ('record', {
        'name': 'songer_record',
        'instruction': f'{header} {header_civil_gov} The issue is: "Did the agency fail to develop an adequate record? For example, if the court was unable to determine what doctrine was used for the decision or unable to determine the basis of the decision. {instructions_issue}',
        'question': 'Did the agency fail to develop an adequate record? For example, if the court was unable to determine what doctrine was used for the decision or unable to determine the basis of the decision.',
        'answer_choices': answer_choices_all,
    }),
    # diversity issues
    # DIVERSE - Did the court conclude that the parties were truly diverse ?
    ('diverse', {
        'name': 'songer_diverse',
        'instruction': f'{header} The issue is: "Did the court conclude that the parties were truly diverse? {instructions_issue}',
        'question': 'Did the court conclude that the parties were truly diverse?',
        'answer_choices': answer_choices_all,
    }),
    # WHLAWS - Did the court's discussion of which state's laws should control their ruling in the case support the position taken by the appellant?
    ('whlaws', {
        'name': 'songer_whlaws',
        'instruction': f'{header} The issue is: "Did the court\'s discussion of which state\'s laws should control their ruling in the case support the position taken by the appellant? {instructions_issue}',
        'question': 'Did the court\'s discussion of which state\'s laws should control their ruling in the case support the position taken by the appellant?',
        'answer_choices': answer_choices_all,
    }),
] 

# Return cases with majority opinion
def filter_cases(dataset):
    for case_ in get_cases_with_maj_opinion(dataset):
        yield case_['caselaw']['id'], case_['songer']

def fill_decision_answer_choices(decision, choices):
    if choices is None or decision in choices:
        return {'target': decision}
    else:
        return None


def create_task_issue(dataset, task_var, task, **save_kwargs):
    # not discussed id is given in case we want to balance one id against the others
    print(f"Processing the Songer documents for the {task['name']} task...")

    if 'answer_choices' not in task:
        choices = None  # the target will be the decision itself
    else:
        choices = task['answer_choices']
        if type(choices) == str:  # file from which to load the answer choices
            if choices.endswith('.txt'):
                # each line is CODE_ID DESCRIPTION, where CODE_ID is an int
                with open(choices, 'r') as f:
                    lines = f.readlines()
                choices = {int(line.split()[0]): line.split()[1] for line in lines}
                # print("!!!! choices", choices)
                task['answer_choices'] = choices
            else:
                raise ValueError(f"Unknown format for answer choices: {choices}")
    
        assert type(choices) == dict, f"Choices should be a dictionary, not {type(choices)}"

    decisions = {} # id_ -> {'input': id_, 'target': decision, 'A': choice_A, 'B': choice_B, ...}
    n_nans = 0
    for id_, case_ in filter_cases(dataset):
        
        if 'get_decision' in task:
            decision = task['get_decision'](case_)
        else:
            decision = case_[task_var]

        if decision != decision:
            decision = None

        if decision is None and (choices is None or 'None' not in choices):
            n_nans += 1
            continue

        decision = int(decision) if decision is not None else 'None'
        decision_dict = fill_decision_answer_choices(decision, choices)

        if decision_dict is None:
            continue

        if 'ignore_targets' in task:
            if decision_dict['target'] in task['ignore_targets']:
                continue

        if 'fill_in' in task:
            if task_var.startswith('direct'):
                code = case_['casetyp' + task_var[-1]]
                if code != code or code == 0:
                    decision_dict['lib_desc'] = ''
                else:
                    desc = direct['lib_desc'][int(str(code)[0])]
                    decision_dict['lib_desc'] = desc
            else:
                for key in task['fill_in']:
                    decision_dict[key] = str(case_[key])

        decisions[id_] = {
            'input': id_,
            **decision_dict
        }

    if 'get_decision' in task:
        del task['get_decision']

    return subsample_and_save_decisions(task, decisions, **save_kwargs)

def get_examples_app_resp_task(dataset, **save_kwargs):
    print(f"Processing the Songer documents for the respondent tasks...")

    seen_ids = set()
    n_examples = {}

    all_tasks = {}
    all_decisions = {}
    for lit_code, cat_code, digit, choice_code, task in build_app_resp_tasks():
        decisions = {}
        for id_, case_ in filter_cases(dataset):
            # check whether the case fits
            code = str(case_[lit_code]).zfill(5)  # five digit code
            if code[0] == str(cat_code):
                if choice_code is None or int(code[digit-2]) == choice_code:
                    if choice_code is None:
                        decision = int(code[digit-1])
                    else:
                        decision = int(code[digit-1:])
                    decision_dict = fill_decision_answer_choices(decision, task['answer_choices'])
                    if decision_dict is not None:
                        decisions[id_] = {
                            'input': id_,
                            **decision_dict,
                        }

        if len(decisions) == 0:
            print(f"No decisions for task {task['name']}")
            continue

        if task['name'].count('_') < 4:
            seen_ids_, n_examples_ = subsample_and_save_decisions(
                task, decisions, **save_kwargs
            )

            seen_ids.update(seen_ids_)
            n_examples[task['name']] = n_examples_
        else:
            all_tasks[task['name']] = task
            all_decisions[task['name']] = decisions
            
    agg_tasks = {}
    agg_decisions = {}
    seen_ids = set()
    for task in all_tasks.values():
        decisions = all_decisions[task['name']]

        if len(decisions) == 0:
            continue

        base_name = task['name'][:-2]
        key = task['name'][-1]

        if base_name not in agg_tasks:
            agg_tasks[base_name] = {
                'name': base_name,
                'instruction': {},
                'question': {},
                'answer_choices': {},
            }
            agg_decisions[base_name] = {}

        agg_tasks[base_name]['question'][key] = task['question']
        agg_tasks[base_name]['instruction'][key] = task['instruction']
        agg_tasks[base_name]['answer_choices'][key] = task['answer_choices']

        # place key in every decision
        decisions = {id_: {**ex, 'key': key} for id_, ex in decisions.items()}

        # check that none of the ids are repeated
        agg_decisions[base_name].update(decisions)

        # print('Aggregated task:', base_name)

    # now we save the aggregated tasks
    for task_name, task in agg_tasks.items():
        decisions = agg_decisions[task_name]
        seen_ids_, n_examples_ = subsample_and_save_decisions(
            task, decisions, **save_kwargs
        )

        seen_ids.update(seen_ids_)
        n_examples[task_name] = n_examples_

    return seen_ids, n_examples

def get_issue_decisions(dataset, is_casetyp1):
    def get_code_2(code_1, code_3):
        if 2 not in case_issues[code_1]:
            return None
        for key, val in case_issues[code_1][3].items():
            if code_3 in val:
                return key
        if not code_3 in [716]:
            raise ValueError(f"Code {code_3} not found in case_issues")


    decisions = {'1': {}, '2': {}, '3': {}}
    for issue_code, issue_text in issue_pre_tasks['geniss']['answer_choices'].items():
        for id_, case_ in filter_cases(dataset):
            key = 'casetyp1' if is_casetyp1 else 'casetyp2'

            if case_[key] != case_[key] or case_[key] == 0:
                continue
            
            # print('case_', case_[key])
            code_1 = int(str(case_[key])[0])
            code_3 = int(str(case_[key]))
            code_2 = get_code_2(code_1, code_3)

            if 2 in case_issues[issue_code] and code_2 is None:
                continue  # coding errors

            if is_casetyp1:
                main_issue = None
            else:
                main_code_1 = int(str(case_['casetyp1'])[0])
                main_code_3 = int(str(case_['casetyp1']))
                main_code_2 = get_code_2(main_code_1, main_code_3)

                main_issue = issue_pre_tasks['geniss']['answer_choices'][main_code_1]
                if main_code_2 is not None:
                    main_issue += f" - {case_issues[main_code_1][2][main_code_2]}"
                    main_issue += f" - {case_issues[main_code_1][3][main_code_2][main_code_3]}"
                else:
                    main_issue += f" - {case_issues[main_code_1][3][main_code_3]}"

            if code_1 != issue_code:
                continue
            
            # the main type of the first issue
            if not is_casetyp1:
                answer_choices = issue_pre_tasks['geniss']['answer_choices']
                decision_dict = fill_decision_answer_choices(code_1, answer_choices)
                if main_issue is not None:
                    decision_dict['mainissue'] = main_issue
                decision_dict['input'] = id_
                decision_dict['answer_choices_key'] = 'geniss'
                decisions['1'][id_] = decision_dict

            if 2 in case_issues[issue_code]:
                answer_choices = case_issues[issue_code][2]
                decision_dict = fill_decision_answer_choices(code_2, answer_choices)
                decision_dict['issue'] = issue_text
                if main_issue is not None:
                    decision_dict['mainissue'] = main_issue
                decision_dict['answer_choices_key'] = f"{issue_code}-2"  # case_issues[issue_code][2]
                decision_dict['input'] = id_
                decisions['2'][id_] = decision_dict
            
            if 3 in case_issues[issue_code]:
                if code_2 is None:
                    answer_choices = case_issues[issue_code][3]
                    key = f"{issue_code}-3"
                else:
                    answer_choices = case_issues[issue_code][3][code_2]
                    key = f"{issue_code}-3-{code_2}"
                decision_dict = fill_decision_answer_choices(code_3, answer_choices)
                if code_2 is None:
                    decision_dict['issue'] = f"{issue_text}"
                else:
                    decision_dict['issue'] = f"{issue_text} - {case_issues[issue_code][2][code_2]}"
                if main_issue is not None:
                    decision_dict['mainissue'] = main_issue
                decision_dict['input'] = id_
                decision_dict['answer_choices_key'] = key
                decisions['3'][id_] = decision_dict
            
    return decisions
    
def get_examples_issues(dataset, **save_kwargs):
    base_task = {
        1: {
            'instruction': f'{header}\n{header_issue}\n' \
                'Your task is to determine the specific issue in the case within the broad category of "{issue}". ',
            'question': 'What is the specific issue in the case within the general category of "{issue}"?',
            'fill_in': ['issue'],
        },
        2: {
            'instruction': f'{header}\n{header_issue}\n' \
            'There are two main issues in this case. The first issue is {mainissue}. ' \
            'Your task is to determine the second issue in the case. ' \
            'Consider the following categories: "criminal" (including appeals of conviction, petitions for post conviction ' \
            'relief, habeas corpus petitions, and other prisoner petitions which ' \
            'challenge the validity of the conviction or the sentence), ' \
            '"civil rights" (excluding First Amendment or due process; also excluding claims of denial of rights in criminal ' \
            'proceeding or claims by prisoners that challenge their conviction or their sentence (e.g., ' \
            'habeas corpus petitions are coded under the criminal category); ' \
            'does include civil suits instituted by both prisoners and callable ' \
            'non-prisoners alleging denial of rights by criminal justice officials), ' \
            '"First Amendment", "due process" (claims in civil cases by persons other than prisoners, ' \
            'does not include due process challenges to government economic regulation), ' \
            '"privacy", "labor relations", "economic activity and regulation", and "miscellaneous".',
            'question': 'What is the second general issue in the case, other than {mainissue}?',
            'fill_in': ['mainissue'],
        },
    }

    def get_task_id(casetyp, digit):
        if casetyp == 1:
            return 1
        if digit == 1:
            return 2
        return 1

    task_collections = {1: {}, 2: {}}
    for casetyp in [1, 2]:
        decisions = get_issue_decisions(dataset, casetyp == 1)
        digits = [2, 3] if casetyp == 1 else [1, 2, 3]
        for digit in digits:
            task_id = get_task_id(casetyp, digit)
            choices = set([ex['answer_choices_key'] for ex in decisions[str(digit)].values()])
            for i, choice in enumerate(choices):
                # get the answer choices
                if choice == 'geniss':
                    answer_choices = issue_pre_tasks['geniss']['answer_choices']
                else:
                    parts = choice.split('-')
                    if len(parts) == 2:
                        answer_choices = case_issues[int(parts[0])][int(parts[1])]
                    elif len(parts) == 3:
                        answer_choices = case_issues[int(parts[0])][int(parts[1])][int(parts[2])]
                    else:
                        raise ValueError(f"Unknown format for answer choices: {choice}")
                    
                # select those decisions that have the same answer choices
                decisions_ = {id_: ex for id_, ex in decisions[str(digit)].items() if ex['answer_choices_key'] == choice}

                if len(decisions_) == 0:
                    continue

                if choice not in task_collections[task_id]:
                    task_collections[task_id][choice] = {
                        'answer_choices': answer_choices,
                        'decisions': decisions_,
                    }
                else:
                    assert answer_choices == task_collections[task_id][choice]['answer_choices'], f"{answer_choices} != {task_collections[task_id][choice]['answer_choices']}"
                    task_collections[task_id][choice]['decisions'].update(decisions_)

    seen_ids = set()
    n_examples = {}

    for task_id, task_collection in task_collections.items():
        for choice, dict_ in task_collection.items():
            decisions = dict_['decisions']

            if len(decisions) == 0:
                continue

            if len(dict_['answer_choices']) <= 1:
                continue

            task = {'name': f'songer_casetyp{task_id}_{choice}', 
                    'answer_choices': dict_['answer_choices'],
                    **base_task[task_id]}
            
            seen_ids_, n_examples_ = subsample_and_save_decisions(
                task=task,
                decisions=decisions,
                **save_kwargs
            )

            seen_ids.update(seen_ids_)
            n_examples[task['name']] = n_examples_
    
    return seen_ids, n_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--train_test_split', type=str, default='splits/splits_songer.json')
    # by default, the task data is subsampled such that the majority class is at most 50% of the task data
    parser.add_argument('--do_not_limit_train', action='store_true')
    parser.add_argument('--do_not_limit_test', action='store_true')
    args = parser.parse_args()

    # Load the data file
    dataset = []
    print("Loading the Supreme Court opinions...")
    with open(args.data_file, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            data = json.loads(line)
            dataset.append(data)

    # Load the train-test split
    with open(args.train_test_split, 'r') as jsonl_file:
        splits = json.load(jsonl_file)

    # Create the save dir if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Get the relevant decisions for each task
    ids = set()
    n_examples = {}

    vanilla_tasks = [
        tasks_general,
        tasks_participants,
        issue_tasks,
        issue_pre_tasks,
        citing_cases,
        {
            'direct1': {'name': 'songer_direct1', **direct},
            'direct2': {'name': 'songer_direct2', **direct}
        },
    ]

    save_kwargs = {
        'splits': splits,
        'save_dir': args.save_dir,
        'limit_train': not args.do_not_limit_train,
        'limit_test': not args.do_not_limit_test,
        'verbose': False,
    }

    for vanilla_task in vanilla_tasks:
        if type(vanilla_task) == dict:
            vanilla_task = vanilla_task.items()
        for task_var, task in tqdm(vanilla_task):
            ids_, n = create_task_issue(dataset, task_var, task, **save_kwargs)

            ids.update(ids_)
            n_examples[task['name']] = n

    ids_, n_examples_ = get_examples_app_resp_task(dataset, **save_kwargs)
    ids.update(ids_)
    n_examples = {**n_examples, **n_examples_}

    ids_, n_examples_ = get_examples_issues(dataset, **save_kwargs)
    ids.update(ids_)
    n_examples = {**n_examples, **n_examples_}

    # sort tasks by number of examples
    n_examples = {k: v for k, v in sorted(n_examples.items(), key=lambda item: item[1])}

    print("Number of examples in each task:")
    cum = 0
    for i, (task, n) in enumerate(n_examples.items()):
        cum += n
        print(f"{i}, {task}: {n:.2f}, cum:{cum:.2f}")

    save_opinions(dataset, ids, args.save_dir, prefix='songer_')

    print("Done!")
