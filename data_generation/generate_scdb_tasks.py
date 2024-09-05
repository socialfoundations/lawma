import os
import json
from tqdm import tqdm

from utils import get_cases_with_maj_opinion, save_opinions, subsample_and_save_decisions

header = "What follows is an opinion from the Supreme Court of the United States."

tasks = (
    ('petitioner', {
        'name': 'sc_petitioner',
        'instruction': f'{header} ' \
            "Your task is to identify the petitioner of the case. The petitioner is the party who petitioned the Supreme Court to review the case. " \
            "This party is variously known as the petitioner or the appellant. " \
            "Characterize the petitioner as the Court's opinion identifies them.\n\n" \
            "Identify the petitioner by the label given to the party in the opinion or judgment of the Court except where the Reports title a party as the \"United States\" or as a named state. " \
            "Textual identification of parties is typically provided prior to Part I of the Court's opinion. The official syllabus, the summary that appears on the title page of the case, may " \
            "be consulted as well. In describing the parties, the Court employs terminology that places them in the context of the specific lawsuit in which they are involved. " \
            "For example, \"employer\" rather than \"business\" in a suit by an employee; as a \"minority,\" \"female,\" or \"minority female\" employee rather than \"employee\" in a " \
            "suit alleging discrimination by an employer.\n\n" \
            "Also note that the Court's characterization of the parties applies whether the petitioner is actually single entity or whether many other persons " \
            "or legal entities have associated themselves with the lawsuit. That is, the presence of the phrase, et al., following the name of a party does not preclude the Court " \
            "from characterizing that party as though it were a single entity. Thus, identify a single petitioner, regardless of how many legal entities were actually involved. " \
            "If a state (or one of its subdivisions) is a party, note only that a state is a party, not the state's name.",
        'question': 'Who is the petitioner of the case?',
        'answer_choices': 'petitioner.txt'
    }),
    ('respondent', {
        'name': 'sc_respondent',
        'instruction': f'{header} ' \
            "Your task is to identify the respondent of the case. The respondent is the party being sued or tried and is also known as the appellee. " \
            "Characterize the respondent as the Court's opinion identifies them.\n\n" \
            "Identify the respondent by the label given to the party in the opinion or judgment of the Court except where the Reports title a party as the \"United States\" or as a named state. " \
            "Textual identification of parties is typically provided prior to Part I of the Court's opinion. The official syllabus, the summary that appears on the title page of the case, may " \
            "be consulted as well. In describing the parties, the Court employs terminology that places them in the context of the specific lawsuit in which they are involved. " \
            "For example, \"employer\" rather than \"business\" in a suit by an employee; as a \"minority,\" \"female,\" or \"minority female\" employee rather than \"employee\" in a " \
            "suit alleging discrimination by an employer.\n\n" \
            "Also note that the Court's characterization of the parties applies whether the respondent is actually single entitiy or whether many other persons " \
            "or legal entities have associated themselves with the lawsuit. That is, the presence of the phrase, et al., following the name of a party does not preclude the Court " \
            "from characterizing that party as though it were a single entity. Thus, identify a single respondent, regardless of how many legal entities were actually involved. " \
            "If a state (or one of its subdivisions) is a party, note only that a state is a party, not the state's name.",
        'question': 'Who is the respondent of the case?',
        'answer_choices': 'respondent.txt'
    }),
    ('petitionerState', {
        'name': 'sc_petitionerstate',
        'instruction': f'{header} Your task is to identify the state associated with the petitioner. ' \
            "If the petitioner is a federal court or federal judge, note the \"state\" as the United States. The same holds for other federal employees or officials.",
        'question': 'What state is associated with the petitioner?',
        'answer_choices': 'states.txt'
    }),
    ('respondentState', {
        'name': 'sc_respondentstate',
        'instruction': f'{header} Your task is to identify the state associated with the respondent. ' \
            "If the respondent is a federal court or federal judge, note the \"state\" as the United States. The same holds for other federal employees or officials.",
        'question': 'What state is associated with the respondent?',
        'answer_choices': 'states.txt'
    }),
    ('jurisdiction', {
        'name': 'sc_jurisdiction',
        'instruction': f'{header} Your task is to identify the manner in which the Court took jurisdiction. ' \
            "The Court uses a variety of means whereby it undertakes to consider cases that it has been petitioned to review. " \
            "The most important ones are the writ of certiorari, the writ of appeal, and for legacy cases the writ of error, appeal, and certification. " \
            "For cases that fall into more than one category, identify the manner in which the court takes jurisdiction on the basis of the writ. " \
            "For example, Marbury v. Madison, 5 U.S. 137 (1803), an original jurisdiction and a mandamus case, should be coded as mandamus rather than original jurisdiction due to the nature of the writ. " \
            "Some legacy cases are \"original\" motions or requests for the Court to take jurisdiction but were heard or filed in another court. " \
            "For example, Ex parte Matthew Addy S.S. & Commerce Corp., 256 U.S. 417 (1921) asked the Court to issue a writ of mandamus to a federal judge. " \
            "Do not code these cases as \"original\" jurisdiction cases but rather on the basis of the writ.",
        'question': 'What is the manner in which the Court took jurisdiction?',
        'answer_choices': 'jurisdiction.txt'
    }),
    ('adminAction', {
        'name': 'sc_adminaction_is',
        'instruction': f"{header} Your task is to identify whether administrative action occurred in the context of the case " \
            "prior to the onset of litigation. The activity may involve an administrative official as well as that of an agency. " \
            "To determine whether administration action occurred in the context of the case, " \
            "consider the material which appears in the summary of the case preceding the Court's opinion and, if necessary, "\
            "those portions of the prevailing opinion headed by a I or II. Action by an agency official is considered to be " \
            "administrative action except when such an official acts to enforce criminal law. If an agency or agency official " \
            "\"denies\" a \"request\" that action be taken, such denials are considered agency action. " \
            "Exclude: " \
            "a \"challenge\" to an unapplied agency rule, regulation, etc.; a request for an injunction or a declaratory judgment " \
            "against agency action which, though anticipated, has not yet occurred; a mere request for an agency to take action " \
            "when there is no evidence that the agency did so; agency or official action to enforce criminal law; the hiring and " \
            "firing of political appointees or the procedures whereby public officials are appointed to office; attorney general " \
            "preclearance actions pertaining to voting; filing fees or nominating petitions required for access to the ballot; " \
            "actions of courts martial; land condemnation suits and quiet title actions instituted in a court; and federally funded " \
            "private nonprofit organizations.",
        'question': 'Did administrative action occur in the context of the case?',
        'answer_choices': {
            'otherwise': 'Yes',
            'nan': 'No',
        },
    }),
    ('adminAction', {
        'name': 'sc_adminaction',
        'instruction': f'{header} Your task is to identify the federal agency involved in the administrative action that occurred prior to the onset of litigation. ' \
            "If the administrative action occurred in a state agency, respond \"State Agency\". Do not code the name of the state. " \
            "The administrative activity may involve an administrative official as well as that of an agency. " \
            "If two federal agencies are mentioned, consider the one whose action more directly bears on the dispute;" \
            "otherwise the agency that acted more recently. If a state and federal agency are mentioned, consider the federal agency. " \
            "Pay particular attention to the material which appears in the summary of the case preceding the Court's opinion and, if necessary, " \
            "those portions of the prevailing opinion headed by a I or II. Action by an agency official is considered to be administrative action " \
            "except when such an official acts to enforce criminal law. If an agency or agency official \"denies\" a \"request\" that action be taken, " \
            "such denials are considered agency action. Exclude: a \"challenge\" to an unapplied agency rule, regulation, etc.; a request for an injunction " \
            "or a declaratory judgment against agency action which, though anticipated, has not yet occurred; a mere request for an agency to take action " \
            "when there is no evidence that the agency did so; agency or official action to enforce criminal law; the hiring and firing of political appointees " \
            "or the procedures whereby public officials are appointed to office; attorney general preclearance actions pertaining to voting; filing fees or nominating " \
            "petitions required for access to the ballot; actions of courts martial; land condemnation suits and quiet title actions instituted in a court; and federally " \
            "funded private nonprofit organizations.",
        'question': 'What is the agency involved in the administrative action?',
        'answer_choices': 'adminaction.txt',
    }),
    ('adminActionState', {
        'name': 'sc_adminactionstate',
        'instruction': f'{header} Your task is to identify the state of the state agency associated with the administrative action that occurred prior to the onset of litigation.',
        'question': 'What is the state of the state agency associated with the administrative action?',
        'answer_choices': 'states.txt',
    }),
    ('threeJudgeFdc', {
        'name': 'sc_threejudgefdc',
        'instruction': f'{header} Your task is to determine whether the case was heard by a three-judge federal district court. ' \
            "Beginning in the early 1900s, Congress required three-judge district courts to hear certain kinds of cases. " \
            "More modern-day legislation has reduced the kinds of lawsuits that must be heard by such a court. " \
            "As a result, the frequency is less for the Burger Court than for the Warren Court, and all but nonexistent for the Rehnquist and Roberts Courts.",
        'question': 'Was the case heard by a three-judge federal district court?',
        'answer_choices': {
            1: 'Yes',
            0: 'No',
        }
    }),
    ('caseOrigin', {
        'name': 'sc_caseorigin',
        'instruction': f'{header} Your task is to identify the court in which the case originated. ' \
            "Focus on the court in which the case originated, not the administrative agency. " \
            "For this reason, if appropiate note the origin court to be a state or federal appellate court rather than a court of first instance (trial court). " \
            "If the case originated in the United States Supreme Court (arose under its original jurisdiction or no other court was involved), " \
            "note the origin as \"United States Supreme Court\". " \
            "If the case originated in a state court, note the origin as \"State Court\". Do not code the name of the state. " \
            "The courts in the District of Columbia present a special case in part because of their complex history. " \
            "Treat local trial (including today's superior court) and appellate courts (including today's DC Court of Appeals) as state courts. " \
            "Consider cases that arise on a petition of habeas corpus and those removed to the federal courts from a state court as originating in the federal, rather than a state, court system. " \
            "A petition for a writ of habeas corpus begins in the federal district court, not the state trial court. " \
            "Identify courts based on the naming conventions of the day. " \
            "Do not differentiate among districts in a state. For example, use \"New York U.S. Circuit for (all) District(s) of New York\" for all the districts in New York.",
        'question': 'What is the court in which the case originated?',
        'answer_choices': 'origin.txt',
    }),
    ('caseOriginState', {
        'name': 'sc_caseoriginstate',
        'instruction': f'{header} Your task is to identify the state of the court in which the case originated. ' \
            "Consider the District of Columbia as a state.",
        'question': 'What is the state of the court in which the case originated?',
        'answer_choices': 'states.txt',
    }),
    ('caseSource', {
        'name': 'sc_casesource',
        'instruction': f'{header} Your task is to identify the court whose decision the Supreme Court reviewed. ' \
            "If the case arose under the Supreme Court's original jurisdiction, note the source as \"United States Supreme Court\". " \
            "If the case arose in a state court, note the source as \"State Supreme Court\", \"State Appellate Court\", or \"State Trial Court\". " \
            "Do not code the name of the state. ",
        'question': 'What is the court whose decision the Supreme Court reviewed?',
        'answer_choices': 'source.txt',
    }),
    ('caseSourceState', {
        'name': 'sc_casesourcestate',
        'instruction': f'{header} Your task is to identify the state or territory of the court whose decision the Supreme Court reviewed. ',
        'question': 'What is the state of the court whose decision the Supreme Court reviewed?',
        'answer_choices': 'states.txt',
    }),
    ('lcDisagreement', {
        'name': 'sc_lcdisagreement',
        'instruction': f'{header} Your task is to identify whether the court opinion mentions that one or more of the members of the court whose decision the Supreme Court reviewed dissented. ' \
                        'Focus on whether there exists any statement to this effect in the opinion, for example "divided," "dissented," "disagreed," "split.". ' \
                        'A reference, without more, to the "majority" or "plurality" does not necessarily evidence dissent (the other judges may have concurred). ' \
                        'If a case arose on habeas corpus, indicate dissent if either the last federal court or the last state court to review the case contained one. ' \
                        'If the highest court with jurisdiction to hear the case declines to do so by a divided vote, indicate dissent. ' \
                        'If the lower court denies an en banc petition by a divided vote and the Supreme Court discusses same, indicate dissent.',
        'question': 'Does the court opinion mention that one or more of the members of the court whose decision the Supreme Court reviewed dissented?',
        'answer_choices': {
            1: 'Yes',
            0: 'No',
        }
    }),
    ('certReason', {
        'name': 'sc_certreason',
        'instruction': f'{header} Your task is to identify the reason, if any, given by the court for granting the petition for certiorari.',
        'question': 'What reason, if any, does the court give for granting the petition for certiorari?',
        'answer_choices': 'certreason.txt',
    }),
    ('lcDisposition', {
        'name': 'sc_lcdisposition',
        'instruction': f'{header} Your task is to determine the treatment the court whose decision the Supreme Court reviewed accorded the decision of the court it reviewed, ' \
                        'that is, whether the court below the Supreme Court (typically a federal court of appeals or a state supreme court) affirmed, reversed, remanded, denied or dismissed ' \
                        'the decision of the court it reviewed (typically a trial court). ' \
                        'Adhere to the language used in the "holding" in the summary of the case on the title page or prior to Part I of the Court\'s opinion. ' \
                        'Exceptions to the literal language are the following: ' \
                        'where the Court overrules the lower court, treat this a petition or motion granted; '
                        'where the court whose decision the Supreme Court is reviewing refuses to enforce or enjoins the decision of the court, tribunal, or agency which it reviewed, treat this as reversed; ' \
                        'where the court whose decision the Supreme Court is reviewing enforces the decision of the court, tribunal, or agency which it reviewed, treat this as affirmed; ' \
                        'where the court whose decision the Supreme Court is reviewing sets aside the decision of the court, tribunal, or agency which it reviewed, treat this as vacated; ' \
                        'if the decision is set aside and remanded, treat it as vacated and remanded.',
        'question': 'What treatment did the court whose decision the Supreme Court reviewed accorded the decision of the court it reviewed?',
        'answer_choices': 'lcdisposition.txt',
    }),
    ('lcDispositionDirection', {
        'name': 'sc_lcdispositiondirection',
        'instruction': f'{header} Your task is to determine whether the decision of the court whose decision the Supreme Court reviewed was itself ' \
            'liberal or conservative. ' \
            'In the context of issues pertaining to criminal procedure, civil rights, First Amendment, due process, privacy, and attorneys, consider liberal to be ' \
            'pro-person accused or convicted of crime, or denied a jury trial, pro-civil liberties or civil rights claimant, ' \
            'especially those exercising less protected civil rights (e.g., homosexuality), pro-child or juvenile, pro-indigent ' \
            'pro-Indian, pro-affirmative action, pro-neutrality in establishment clause cases, pro-female in abortion, pro-underdog, anti-slavery, incorporation of foreign territories ' \
            'anti-government in the context of due process, except for takings clause cases where a pro-government, ' \
            'anti-owner vote is considered liberal except in criminal forfeiture cases or those where the taking is pro-business ' \
            'violation of due process by exercising jurisdiction over nonresident, pro-attorney or governmental official in non-liability cases, pro-accountability and/or anti-corruption in campaign spending ' \
            'pro-privacy vis-a-vis the 1st Amendment where the privacy invaded is that of mental incompetents, pro-disclosure in Freedom of Information Act issues except for employment and student records. ' \
            'In the context of issues pertaining to unions and economic activity, consider liberal to be ' \
            'pro-union except in union antitrust where liberal = pro-competition, pro-government, anti-business ' \
            'anti-employer, pro-competition, pro-injured person, pro-indigent, pro-small business vis-a-vis large business ' \
            'pro-state/anti-business in state tax cases, pro-debtor, pro-bankrupt, pro-Indian, pro-environmental protection, pro-economic underdog ' \
            'pro-consumer, pro-accountability in governmental corruption, pro-original grantee, purchaser, or occupant in state and territorial land claims ' \
            'anti-union member or employee vis-a-vis union, anti-union in union antitrust, anti-union in union or closed shop, pro-trial in arbitration. ' \
            'In the context of issues pertaining to judicial power, consider liberal to be pro-exercise of judicial power, pro-judicial "activism", pro-judicial review of administrative action. ' \
            'In the context of issues pertaining to federalism, consider liberal to be pro-federal power, pro-executive power in executive/congressional disputes, anti-state. ' \
            'In the context of issues pertaining to federal taxation, consider liberal to be pro-United States and conservative pro-taxpayer. ' \
            'In miscellaneous, consider conservative the incorporation of foreign territories and executive authority vis-a-vis congress or the states or judcial authority vis-a-vis state or federal legislative authority, ' \
            'and consider liberal legislative veto. ' \
            'The lower court\'s decision direction is unspecifiable if the manner in which the Supreme Court took jurisdiction is original or certification; or if ' \
            "the direction of the Supreme Court's decision is unspecifiable and the main issue pertains to private law or interstate relations",
        'question': 'What is the ideological direction of the decision reviewed by the Supreme Court?',
        'answer_choices': {
            1: 'Conservative',
            2: 'Liberal',
            3: 'Unspeciﬁable'
        }
    }),
    ('issueArea', {
        'name': 'sc_issuearea',
        'instruction': f'{header} Your task is to determine the issue area of the Court\'s decision. ' \
            'Determine the issue area on the basis of the Court\'s own statements as to what the case is about. ' \
            'Focus on the subject matter of the controversy rather than its legal basis. ' \
            'In specifying the issue in a legacy case, choose the one that best accords with what today\'s Court would consider it to be. ' \
            'Choose among the following issue areas: ' \
            '"Criminal Procedure" encompasses the rights of persons accused of crime, except for the due process rights of prisoners. ' \
            '"Civil rights" includes non-First Amendment freedom cases which pertain to classifications based on race ' \
            '(including American Indians), age, indigency, voting, residency, military or handicapped status, gender, and alienage. ' \
            '"First Amendment encompasses the scope of this constitutional provision, but do note that it need not ' \
            'involve the interpretation and application of a provision of the First Amendment. ' \
            'For example, if the case only construe a precedent, or the reviewability of a claim based on the First Amendment, ' \
            'or the scope of an administrative rule or regulation that impacts the exercise of First Amendment freedoms. ' \
            '"Due process" is limited to non-criminal guarantees. ' \
            '"Privacy" concerns libel, comity, abortion, contraceptives, right to die, and Freedom of Information Act and related federal or state statutes or regulations. ' \
            '"Attorneys" includes attorneys\' compensation and licenses, along with trhose of governmental officials and employees. ' \
            '"Unions" encompass those issues involving labor union activity. ' \
            '"Economic activity" is largely commercial and business related; it includes tort actions and employee actions vis-a-vis employers. ' \
            '"Judicial power" concerns the exercise of the judiciary\'s own power. ' \
            '"Federalism" pertains to conflicts and other relationships between the federal government and the states, except for those between the federal and state courts. ' \
            '"Federal taxation" concerns the Internal Revenue Code and related statutes. ' \
            '"Private law" relates to disputes between private persons involving real and personal property, ' \
            'contracts, evidence, civil procedure, torts, wills and trusts, and commercial transactions. Prior to ' \
            'the passage of the Judges\' Bill of 1925 much of the Court\'s cases concerned such issues. ' \
            'Use "Miscellaneous" for legislative veto and executive authority vis-a-vis congress or the states.',
        'question': 'What is the issue area of the decision?',
        'answer_choices': 'issuearea.txt',
    }),
    ('issue', {
        'name': 'sc_issue',
        'instruction': f'{header} Your task is to determine the issue of the Court\'s decision. ' \
            'Determine the issue of the case on the basis of the Court\'s own statements as to what the case is about. ' \
            'Focus on the subject matter of the controversy rather than its legal basis.',
        'question': 'What is the issue of the decision?',
        'answer_choices': 'issue.txt',
    }),
    ('decisionDirection', {
        'name': 'sc_decisiondirection',
        'instruction': f'{header} Your task is to determine the ideological "direction" of the decision ("liberal", "conservative", or "unspecifiable"). ' \
                        'Use "unspecifiable" if the issue does not lend itself to a liberal or conservative description (e.g., a boundary dispute between two states, real property, wills and estates), ' \
                        'or because no convention exists as to which is the liberal side and which is the conservative side (e.g., the legislative veto). '
                        'Specification of the ideological direction comports with conventional usage. ' \
                        'In the context of issues pertaining to criminal procedure, civil rights, First Amendment, due process, privacy, and attorneys, consider liberal to be ' \
                        'pro-person accused or convicted of crime, or denied a jury trial, pro-civil liberties or civil rights claimant, ' \
                        'especially those exercising less protected civil rights (e.g., homosexuality), pro-child or juvenile, pro-indigent ' \
                        'pro-Indian, pro-affirmative action, pro-neutrality in establishment clause cases, pro-female in abortion, pro-underdog, anti-slavery, incorporation of foreign territories ' \
                        'anti-government in the context of due process, except for takings clause cases where a pro-government, ' \
                        'anti-owner vote is considered liberal except in criminal forfeiture cases or those where the taking is pro-business ' \
                        'violation of due process by exercising jurisdiction over nonresident, pro-attorney or governmental official in non-liability cases, pro-accountability and/or anti-corruption in campaign spending ' \
                        'pro-privacy vis-a-vis the 1st Amendment where the privacy invaded is that of mental incompetents, pro-disclosure in Freedom of Information Act issues except for employment and student records. ' \
                        'In the context of issues pertaining to unions and economic activity, consider liberal to be ' \
                        'pro-union except in union antitrust where liberal = pro-competition, pro-government, anti-business ' \
                        'anti-employer, pro-competition, pro-injured person, pro-indigent, pro-small business vis-a-vis large business ' \
                        'pro-state/anti-business in state tax cases, pro-debtor, pro-bankrupt, pro-Indian, pro-environmental protection, pro-economic underdog ' \
                        'pro-consumer, pro-accountability in governmental corruption, pro-original grantee, purchaser, or occupant in state and territorial land claims ' \
                        'anti-union member or employee vis-a-vis union, anti-union in union antitrust, anti-union in union or closed shop, pro-trial in arbitration. ' \
                        'In the context of issues pertaining to judicial power, consider liberal to be pro-exercise of judicial power, pro-judicial "activism", pro-judicial review of administrative action. ' \
                        'In the context of issues pertaining to federalism, consider liberal to be pro-federal power, pro-executive power in executive/congressional disputes, anti-state. ' \
                        'In the context of issues pertaining to federal taxation, consider liberal to be pro-United States and conservative pro-taxpayer. ' \
                        'In miscellaneous, consider conservative the incorporation of foreign territories and executive authority vis-a-vis congress or the states or judcial authority vis-a-vis state or federal legislative authority, ' \
                        'and consider liberal legislative veto. ' \
                        'In interstate relations and private law issues, consider unspecifiable in all cases.',
        'question': 'What is the ideological direction of the decision?',
        'answer_choices': {
            1: 'Conservative',
            2: 'Liberal',
            3: 'Unspeciﬁable'
        }
    }),
    ('authorityDecision1', {  # multilabel
        'name': 'sc_authoritydecision',
        'instruction': f'{header} Your task is to determine the bases on which the Supreme Court rested its decision with regard to ' \
            'the legal provision that the Court considered in the case. ' \
            'Consider "judicial review (national level)" if the majority determined the constitutionality of some action taken by some unit ' \
            'or official of the federal government, including an interstate compact. ' \
            'Consider "judicial review (state level)" if the majority determined the constitutionality of some action taken by some ' \
            'unit or official of a state or local government. ' \
            'Consider "statutory construction" for cases where the majority interpret a federal statute, treaty, or court rule; ' \
            'if the Court interprets a federal statute governing the powers or jurisdiction of a federal court; ' \
            'if the Court construes a state law as incompatible with a federal law; or if an administrative official interprets a federal statute. ' \
            'Do not consider "statutory construction" where an administrative agency or official acts "pursuant to" a statute, unless ' \
            'the Court interprets the statute to determine if administrative action is proper. ' \
            'Consider \"interpretation of administrative regulation or rule, or executive order\" if the majority treats federal administrative ' \
            'action in arriving at its decision.'
            'Consider "diversity jurisdiction" if the majority said in approximately so many words that under its diversity jurisdiction it is interpreting state law. ' \
            'Consider \"federal common law\" if the majority indicate that it used a judge-made "doctrine" or "rule; ' \
            'if the Court without more merely specifies the disposition the Court has made of the case and cites one or more of its own previously decided cases ' \
            'unless the citation is qualified by the word "see."; if the case concerns admiralty or maritime law, or some other aspect of the law of ' \
            'nations other than a treaty; ' \
            'if the case concerns the retroactive application of a constitutional provision or a previous decision of the Court; ' \
            'if the case concerns an exclusionary rule, the harmless error rule (though not the statute), the abstention doctrine, comity, res judicata, or collateral estoppel; ' \
            'or if the case concerns a "rule" or "doctrine" that is not specified as related to or connected with a constitutional or statutory provision. ' \
            'Consider "Supreme Court supervision of lower federal or state courts or original jurisdiction" otherwise (i.e., the residual code); ' \
            'for issues pertaining to non-statutorily based Judicial Power topics; for ' \
            'cases arising under the Court\'s original jurisdiction; ' \
            'in cases in which the Court denied or dismissed the petition for review or where the ' \
            'decision of a lower court is affirmed by a tie vote; or in workers\' compensation litigation involving ' \
            'statutory interpretation and, in addition, a discussion of jury determination and/or the sufficiency of the evidence.',
        'question': 'What is the basis of the Supreme Court\'s decision?',
        'multilabel': 'authorityDecision2',
        'answer_choices': 'authority.txt',
    }),
    ('decisionType', {
        'name': 'sc_decisiontype',
        'instruction': f'{header} Your task is to identify the type of decision made by the court among the following: ' \
                        'Consider "opinion of the court (orally argued)" if the court decided the case by a signed opinion and the case was orally argued. ' \
                        'For the 1791-1945 terms, the case need not be orally argued, but a justice must be listed as delivering the opinion of the Court. ' \
                        'Consider "per curiam (no oral argument)" if the court decided the case with an opinion but without hearing oral arguments. ' \
                        'For the 1791-1945 terms, the Court (or reporter) need not use the term "per curiam" but rather "The Court [said],"' \
                        '"By the Court," or "By direction of the Court." ' \
                        'Consider "decrees" in the infrequent type of decisions where the justices will typically appoint a special master to take testimony and ' \
                        'render a report, the bulk of which generally becomes the Court\'s decision. This type of decision ' \
                        'usually arises under the Court\'s original jurisdiction and involves state boundary disputes. ' \
                        'Consider "equally divided vote" for cases decided by an equally divided vote, for example when a justice fails to ' \
                        'participate in a case or when the Court has a vacancy. ' \
                        'Consider "per curiam (orally argued)" if no individual justice\'s name appears as author of the Court\'s opinion and the case was orally argued. ' \
                        'Consider "judgment of the Court (orally argued)" for formally decided cases (decided the case by a signed opinion) ' \
                        'where less than a majority of the participating justices agree with the opinion produced by the justice assigned to write the Court\'s opinion.',
        'question': 'What type of decision did the court make?',
        'answer_choices': 'decisiontype.txt',
    }),
    ('declarationUncon', {
        'name': 'sc_declarationuncon',
        'instruction': f'{header} Your task is to indentify whether the Court declared unconstitutional an act of Congress; a state or territorial statute, regulation, or constitutional provision; or a municipal or other local ordinance. ' \
                        'Note that the Court need not necessarily specify in many words that a law has been declared unconstitutional. ' \
                        'Where federal law pre-empts a state statute or a local ordinance, unconstitutionality does not result unless the Court\'s opinion so states. ' \
                        'Nor are administrative regulations the subject of declarations of unconstitutionality unless the declaration also applies to the law on which it is based. ' \
                        'Also excluded are federal or state court-made rules. ',
        'question': 'Did the Court declare unconstitutional an act of Congress; a state or territorial statute, regulation, or constitutional provision; or a municipal or other local ordinance?',
        'answer_choices': {
            1: 'No declaration of unconstitutionality',
            2: 'Act of Congress declared unconstitutional',
            3: 'State or territorial law, regulation, or constitutional provision unconstitutional',
            4: 'Municipal or other local ordinance unconstitutional',
        }
    }),
    ('caseDisposition', {
        'name': 'sc_casedisposition',
        'instruction': f'{header} Your task is to identify the disposition of the case, that is, the treatment the Supreme Court accorded the court whose decision it reviewed. ' \
                'The information relevant to this variable may be found near the end of the summary that begins ' \
                'on the title page of each case, or preferably at the very end of the opinion of the Court. ' \
                'For cases in which the Court granted a motion to dismiss, consider "petition denied or appeal dismissed". ' \
                'There is "no disposition" if the Court denied a motion to dismiss.',
        'question': 'What is the disposition of the case, that is, the treatment the Supreme Court accorded the court whose decision it reviewed?',
        'answer_choices': 'casedisposition.txt',
    }),
    ('partyWinning', {
        'name': 'sc_partywinning',
        'instruction': f'{header} Your task is to identify whether the petitioning party (i.e., the plaintiff or the appellant) emerged victorious. ' \
                        'The victory the Supreme Court provided the petitioning party may not have been total and complete (e.g., ' \
                        'by vacating and remanding the matter rather than an unequivocal reversal), but the disposition is nonetheless a favorable one. ' \
                        'Consider that the petitioning party lost if the Supreme Court affirmed or dismissed the case, or denied the petition. ' \
                        'Consider that the petitioning party won in part or in full if the Supreme Court reversed, reversed and remanded, vacated and remanded, affirmed and reversed in part, affirmed and reversed in part and remanded, or vacated the case.',
        'question':     'Consider that the petitioning party lost if the Supreme Court affirmed or dismissed the case, or denied the petition. ' \
                        'Consider that the petitioning party won in part or in full if the Supreme Court reversed, reversed and remanded, vacated and remanded, affirmed and reversed in part, affirmed and reversed in part and remanded, or vacated the case. ' \
                        'Did the petitioning win the case?',
        'answer_choices': {
            1: 'Yes',
            0: 'No',
        }
    }),
    ('precedentAlteration', {
        'name': 'sc_precedentalteration',
        'instruction': f'{header} Your task is to identify whether the opinion effectively says that the decision in this case "overruled" one or more of the Court\'s own precedents.' \
            ' Alteration also extends to language in the majority opinion that states that a precedent of the Supreme Court has been "disapproved," or is "no longer good law". ' \
            'Note, however, that alteration does not apply to cases in which the Court "distinguishes" a precedent.',
        'question': 'Did the the decision of the court overrule one or more of the Court\'s own precedents?',
        'answer_choices': {
            1: 'Yes',
            0: 'No',
        }
    }),
)

def get_valid_cases(dataset, task_var, choices, multilabel=None):
    for case in get_cases_with_maj_opinion(dataset):

        decision = case['sc_db'][task_var]
        if decision != decision:
            decision = 'nan'
        else:
            decision = int(decision)

        if decision not in choices:
            if 'otherwise' in choices:
                decision = 'otherwise'
            else:
                continue
        
        if multilabel:
            decision2 = case['sc_db'][multilabel]
            if not (decision2 != decision2):  # not nan
                decision2 = int(decision2)
                decision = [decision, decision2]

        yield case['caselaw']['id'], decision, case['sc_db']

def get_answer_choices(task):
    choices = task['answer_choices']
    if type(choices) == str:  # file from which to load the answer choices
        if choices.endswith('.txt'):
            # each line is CODE_ID DESCRIPTION, where CODE_ID is an int or 'nan'
            with open(f'sc_codes/{choices}', 'r') as f:
                lines = f.readlines()
            choices = {line.split()[0]: ' '.join(line.split()[1:]) for line in lines}
            choices = {int(k) if k != 'nan' else k: v for k, v in choices.items()}
            task['answer_choices'] = choices
        else:
            raise ValueError(f"Unknown format for answer choices: {choices}")
    assert type(choices) == dict, f"Choices should be a dictionary, not {type(choices)}"

    return choices

def create_task(dataset, task_var, task, **kwargs):
    decisions = {}
    print(f"Processing the Supreme Court opinions for the {task['name']} task...")

    answer_choices = get_answer_choices(task)

    decisions = {}
    multilabel = task['multilabel'] if 'multilabel' in task else None

    for id_, decision, case_ in get_valid_cases(dataset, task_var, answer_choices, multilabel):
        decisions[id_] = {'input': id_, 'target': decision}
        
    return post_process_task(task, decisions, answer_choices, **kwargs)


def create_issue_tasks(dataset, task_var, task, **kwargs):
    def get_issue_choices(all_choices, issue_area):
        issue_choices = {}
        for k in all_choices:
            k_str = str(k)
            if len(k_str) < 6:  # should be 6 digits
                k_str = '0' * (6 - len(k_str)) + k_str
            
            if int(k_str[:2]) == int(issue_area):
                issue_choices[int(k_str)] = all_choices[k]

        print(f"{len(issue_choices)} issue choices for issue area {issue_area}")
        return issue_choices

    issue_areas = list(range(1, 15))
    decisions = {i: {} for i in issue_areas}

    answer_choices = get_answer_choices(task)
    multilabel = task['multilabel'] if 'multilabel' in task else None
    for id_, decision, case_ in get_valid_cases(dataset, task_var, answer_choices, multilabel):
        decisions[case_['issueArea']][id_] = {'input': id_, 'target': decision}

    task_names = {i: f"{task['name']}_{i}" for i in issue_areas}
    answer_choices = {i: get_issue_choices(answer_choices, i) for i in issue_areas}

    n_examples = 0
    ids = set()
    for issue_area in list(range(1, 13)):  # last two only have 23 and 4 examples
        task['name'] = task_names[issue_area]
        print(f"Processing the Supreme Court opinions for the {task['name']} task...")
        ids_, n = post_process_task(task, decisions[issue_area], answer_choices[issue_area], **kwargs)
        n_examples += n
        ids = ids.union(ids_)
    
    return ids, n_examples


def post_process_task(task, decisions, answer_choices, **kwargs):
    # Deal with nan in answer choices -- convert them to int
    if 'nan' in answer_choices:
        int_choices = [k for k in answer_choices.keys() if type(k) == int]
        if len(int_choices) == 0:
            nan_code = 0
        else:
            nan_code = max([k for k in answer_choices.keys() if type(k) == int]) + 1

        answer_choices[nan_code] = answer_choices['nan']
        del answer_choices['nan']
        for id_ in decisions:
            if decisions[id_]['target'] == 'nan':
                decisions[id_]['target'] = nan_code

    # deal with 'otherwise' in answer choices -- convert them to int
    if 'otherwise' in answer_choices:
        otherwise_code = max([k for k in answer_choices.keys() if type(k) == int]) + 1
        answer_choices[otherwise_code] = answer_choices['otherwise']
        del answer_choices['otherwise']
        for id_ in decisions:
            if decisions[id_]['target'] == 'otherwise':
                decisions[id_]['target'] = otherwise_code

    return subsample_and_save_decisions(task, decisions, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--train_test_split', type=str, default='splits/splits_scdb.json')
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

    kwargs = {
        'splits': splits,
        'save_dir': args.save_dir,
        'limit_train': not args.do_not_limit_train,
        'limit_test': not args.do_not_limit_test,
    }

    # Create a dataset for each task
    n_examples = 0
    ids = set()  # to keep track of what opinions to save
    for task_var, task in tasks:
        if task_var == 'issue':
            ids_, n = create_issue_tasks(dataset, task_var, task, **kwargs)
        else:
            ids_, n = create_task(dataset, task_var, task, **kwargs)
        ids = ids.union(ids_)
        n_examples += n

    print(f"Total number of examples: {n_examples}")
    
    # Save the opinions corresponding to each id in the dataset
    save_opinions(dataset, ids, args.save_dir, prefix='sc_')

    print("Done!")
