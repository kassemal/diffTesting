"""
Methods that read 'Internet Usage' dataset and the related generalization trees. 
"""

#!/usr/bin/env python
# coding=utf-8

import utils.utility as ul
import pickle
from pulp import *

#Some remarkable attributes numbers:
#2 Age (index = 1): Not-Say, 41, 28, 25, 17, 55, 53, 32, 65, 49, 27, 33, 44, 63, 18, 30, 34, 39, 60, 26, 35, 50, 40, 29, 23, 19, 45, 31, 59, 37, 54, 24, 36, 20, 
                                #48, 69, 42, 21, 47, 43, 38, 22, 5, 57, 68, 52, 15, 51, 16, 56, 46, 64, 73, 62, 80, 58, 61, 7, 14, 67, 70, 72, 76, 71, 66, 78, 13, 79, 77, 9, 
                                #6, 75, 74, 8, 12, 11, 10 (77 ->[5,80])
#12 Country: Texas, Florida, Illinois, Ontario, Washington, Oklahoma, California, Oregon, Alberta, Kentucky, 
                       #North-Carolina, Georgia, Pennsylvania, Indiana, Virginia, Australia, Michigan, Ohio, Connecticut, Rhode-Island, 
                       #New-York, United-Kingdom, Massachusetts, Saskatchewan, Idaho, Wisconsin, New-Jersey, Italy, South-Dakota, Louisiana, 
                       #Vermont, Missouri, Mississippi, Netherlands, Kansas, Alaska, Minnesota, Colorado, Maryland, Utah, Nevada, Washington-DC, 
                       #Wyoming, Arizona, New-Hampshire, South-Carolina, Delaware, Tennessee, Sweden, Afghanistan, Iowa, British-Columbia, Arkansas, 
                       #Montana, France, Alabama, Kuwait, Finland, Switzerland, New-Zealand, Belgium, China, Spain, Manitoba, Maine, Hong-Kong, Nebraska, 
                       #Germany, West-Virginia, Brazil, New-Brunswick, Quebec, Other, Colombia, Hawaii, Japan, South-Africa, Portugal, New Mexico, Austria, India, 
                       #Namibia, Argentina, Israel, Ireland, Nova-Scotia, Thailand, Singapore, Taiwan, North-Dakota, Philippines, Turkey, Venezuela, Denmark, Malaysis, 
                       #Greece, Norway, South-Korea, Oman, Bhutan, Iceland, Czech, Prince Edward Island, Chile, Panama, Newfoundland, Hungary, Egypt, Russia, Ecuador, 
                       #Croatia, Poland, Morocco, Puerto-Rico, Costa-Rica, Dominican-Republic, Jamaica, Yukon, Northwest Territories, Netherlands Antilles, Kenya, 
                       #Sri-Lanka, Indonesia, Romania, Armenia, Algeria, Tunisia, Nicaragua, Burundi (129)
#19 Education Attainment: Masters, Some-College, College, High-School, Professional, Grammar, Special, Doctoral, Other (9)
#21 Gender	: Male, Female (2)
#22 Household Income: Over-$100, under-$10, $50-74, $75-99, Not-Say, $30-39, $20-29, $10-19, $40-49 (9)
#32 Major Geographical Location	
#33 Major Occupation: Professional, Education, Computer, Other, Management (5)
#34 Marital Status: Married, Single, Other, Divorced, Separated, Widowed, Not-Say (7)
#36 Opinions on Censorship	
#37 Primary Computing Platform	
#38 Primary Language: English, Spanish, Italian, Dutch, american, Swedish, Russian, French, Chinese, Finnish, Englishmalay, German, URDU, Portuguese, 
                                    #Slovenian, Bengali, Kannada, EnglishAmerican-Sign-Language, Hebrew, afrikaans, EnglishCajun, GermanSwiss-German, Afrikaans, 
                                    #Korean, Englishpig-latin, EnglishFILIPINO, Turkish, Japanese, Not-Say, EnglishAustralian, Hindi, Danish, english, Greek, Englishand-also-Spanish, 
                                    #SOUTHERN, Norwegian, swedish, danish, bilingual-in-Spanish-and-english, Arabic, Englishestonian, Japanesenot-really, EnglishPolish, 
                                    #Englishtagalog, Icelandic, Tamil, Englishczech, American-Sign-Language, icelandic, Tagalog, EnglishTagalog, Englishsouthern-english, 
                                    #scottish-gaelic, bosnian, swahili, Serbian, Filipino, both-english+italian, Polish, Hungarian, EnglishMandarin-Chinese, 
                                    #norwegian, Bulgarian, EnglishTamil, croatian, turkish, korean, SpanishNorwegian, ENGLISH, EnglishSinhalese, Englishfrench, EnglishAmerican, 
                                    #EnglishNew-Zealand, EnglishAmerican-Southern, EnglishSPANISH, American, Croatian, finnish, Macedonian, Englishaustralian, Englishhawaiian, 
                                    #EnglishFilipino, Urdu, Hindienglish, Bahasa-Malaysia, Malay, Telugu, Indonesian, Englishitalian-french-german, cherakee, Not-Saymardesouquot, 
                                    #Danisk, swedishn, Norvegian, Romanian, Swiss-German, EnglishHungarian, EnglishAfrikaans, EnglishGreek, canadian-english, EnglishAustralian-English, 
                                    #Frenchfrench, spanish, australian, SpanishENGLISH, Swiss-German, maltese, Lithaunian, united-states-of-america-english-with-southern-accent, Germanaustrianic, 
                                    #Bengalidanish, EnglishTurkish, Thai, EnglishEbonics, hebrew (116) 
#39 Primary Place of WWW Access	
#40 Race: White, Hispanic, Indigenous, Not-Say, Other, Latino, Black, Asian (8)
#60 Registered to Vote	
#61 Sexual Preference	
#62 Web Ordering	
#63 Web Page Creation		
#70 Willingness to Pay Fees	
#71 Years on Internet: 1-3-yr, Under-6-mo, 4-6-yr, 6-12-mo, Over-7-yr (5)

#Attribute names as ordered in 'data/internet.data' file 
ATT_NAMES = ['actual_time', 'age', 'community_building', 'community_membership_family',
                                  'community_membership_hobbies', 'community_membership_none', 'community_membership_other',
                                  'community_membership_political', 'community_membership_professional', 'community_membership_religious',
                                  'community_membership_support',  'country', 'disability_cognitive', 'disability_hearing', 'disability_motor', 
                                  'disability_not_impaired', 'disability_not_say', 'disability_vision', 'education_attainment', 'falsification_of_information', 
                                  'gender', 'household_income', 'how_you_heard_about_survey_banner', 'how_you_heard_about_survey_friend',  
                                  'how_you_heard_about_survey_mailing_list',  'how_you_heard_about_survey_others', 'how_you_heard_about_printed_media', 
                                  'how_you_heard_about_survey_remebered',  'how_you_heard_about_survey_search_engine', 'how_you_heard_about_usenet_news', 
                                  'how_you_heard_about_www_page', 'major_geographical_location', 'major_occupation', 'marital_status',  
                                  'most_import_issue_facing_the_internet', 'opinions_on_censorship', 'primary_computing_platform',  'primary_language',  
                                  'primary_place_of_www_access', 'race', 'not_purchasing_bad_experience', 'not_purchasing_bad_press', 'not_purchasing_cant_find',  
                                  'not_purchasing_company_policy', 'not_purchasing_easier_locally', 'not_purchasing_enough_info', 'not_purchasing_judge_quality ',  
                                  'not_purchasing_never_tried ', 'not_purchasing_no_credit', 'not_purchasing_not_applicable', 'not_purchasing_not_option', 
                                  'not_purchasing_other', 'not_purchasing_prefer_people', 'not_purchasing_privacy', 'not_purchasing_receipt', 'not_purchasing_security', 
                                  'not_purchasing_too_complicated', 'not_purchasing_uncomfortable', 'not_purchasing_unfamiliar_vendor',  'registered_to_vote', 
                                  'sexual_preference', 'web_ordering', 'web_page_creation',  'who_pays_for_access_dont_know',  'who_pays_for_access_other', 
                                  'who_pays_for_access_parents', 'who_pays_for_access_school', 'who_pays_for_access_self', 
                                  'who_pays_for_access_work', 'willingness_to_pay_fees',  'years_on_internet',  'pseudonym']

#'False' means that the attribute values are continuous or ordinal  
#'True' means that the attribute is categorical 
CATEGORY = [True, False, True,  False, False, False, False, False, False, False, False, True,  
False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, 
False, False, False, True, True, True, True,  False,  True, True, True, True, False, False, False, False,  
False, False, False, False, False, False, False, False, False, False, False,  False, False, False, False, 
True, True, True, True, False, False, False,  False, False, False,  True, True,  True]


def read():
    """
    read internet usage data from 'data/internet.data'
    """
    #initialize
    nb_attributes = len(ATT_NAMES)
    data, numeric_dict = [], []
    for j in range(nb_attributes):
        if CATEGORY[j] is False:
          numeric_dict.append(dict()) #dictionary for continuous attributes
    #read data
    data_file = open('data/internet.data', 'rU')
    for line in data_file:      
        line = line.strip()
        temp = line.split('\t')
        # remove all the records where 'age' takes the value 'Not-Say' 
        # Only 9799 records will remain 
        if temp[1] == 'Not-Say':
            continue
        #replace missing entries by '?' 
        for j in range(len(temp)):
            if temp[j] == '':
                temp[j] = '?'
        #verify that the number of attributes in each record is 72
        if len(temp) == 72:
            data.append(temp)
        else:
          continue
        #keep a dictionary of continuous attributes
        index = 0
        for j in range(nb_attributes):
            if CATEGORY[j] is False:
                try:
                    numeric_dict[index][temp[j]] += 1
                except:
                    numeric_dict[index][temp[j]] = 1
                index += 1
    # pickle numeric attributes and get NumRange
    index = 0
    for j in range(nb_attributes):
      if CATEGORY[j] is False:
        static_file = open('data/internet_' + ATT_NAMES[j] + '_static.pickle', 'wb')  
        sort_value = list(numeric_dict[index].keys())
        sort_value.sort(cmp=ul.cmp_str)
        pickle.dump((numeric_dict[index], sort_value), static_file)
        static_file.close()
        index += 1
    return data


  