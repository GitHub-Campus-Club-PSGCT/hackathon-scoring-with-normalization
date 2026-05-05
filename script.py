from random import randint
teamslist = ["Byte Me","TriDroplets","DRIZZLE DEFENDERS","DrizzleWorks","MENTATS","DriftCoders","TECH HACKERS","Byte Builders   ","The Elite","TRY_HARD","CADS Team (Cognitive AI Decision Systems)","HighFive","Model_Minds","Team 404","Init to Win it","Tech Strikers","HackAura","We 3","Team Petrichor","Spartans","Null Pointers","InSync","NodeaX","TEAM_INNOVATORS","DrizzleDefenders","Packet Ninjas","DDS","Storm breakers","INCREDIBLES","AGROVISION","Voidsec","Debug Divas","CyberSafe","Digital Drizzle","ak1","dedSec","ForenX","High-On-Code","Dream Weavers","Thooral ML Titans"]
teamidset = set()
teamname_withteamid = []
for team in teamslist:
    while True:
        teamid = randint(1000,9999)
        if teamid not in teamidset:
            teamidset.add(teamid)
            teamname_withteamid.append({"id": "THOORAL"+str(teamid), "name": team})
            break

for team in teamname_withteamid:
    print(team)
