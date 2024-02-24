import discord
import os
import random
import json
from dotenv import load_dotenv
from discord import app_commands
from sentence_transformers import SentenceTransformer, util
import pickle

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

token = os.getenv("token")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Ex: {question1: answer1, ...}
with open("questions.json", encoding="utf8") as f:
    questions = json.load(f)

extrasolar_images = {
    'images/ExtrasolarObjects/Kepler-186b-1.png': "Kepler-186b",
    'images/ExtrasolarObjects/Kepler-186b-2.png': "Kepler-186b",
    'images/ExtrasolarObjects/Kepler-186c.png': 'Kepler-186c',
    'images/ExtrasolarObjects/Kepler-186d.png': 'Kepler-186d',
    'images/ExtrasolarObjects/Kepler-186e.png': 'Kepler-186e',
    'images/ExtrasolarObjects/Kepler-186f.jpg': 'Kepler-186f',
    'images/ExtrasolarObjects/PC-B.jpg': 'Proxima Centauri B',
    'images/ExtrasolarObjects/PC-C.jpg': 'Proxima Centauri C',
    'images/ExtrasolarObjects/PC-D.jpg': 'Proxima Centauri D',
    'images/ExtrasolarObjects/TOI-700B.png': "TOI-700B",
    'images/ExtrasolarObjects/TOI-700C.png': "TOI-700C",
    'images/ExtrasolarObjects/TOI-700D.jpg': "TOI-700D",
    'images/ExtrasolarObjects/TOI-700E.jpg': "TOI-700E",
    'images/ExtrasolarObjects/trappist1b.png': "TRAPPIST-1b",
    'images/ExtrasolarObjects/trappist1c.png': "TRAPPIST-1c",
    'images/ExtrasolarObjects/trappist1d.png': "TRAPPIST-1d",
    'images/ExtrasolarObjects/trappist1e.png': "TRAPPIST-1e",
    'images/ExtrasolarObjects/trappist1f.png': "TRAPPIST-1f",
    'images/ExtrasolarObjects/trappist1g.png': "TRAPPIST-1g",
    'images/ExtrasolarObjects/trappist1h.png': "TRAPPIST-1h"
}

spacecrafts_images = {
    'images/Spacecrafts/cassini.jpeg': 'Cassini',
    'images/Spacecrafts/curiosity.jpeg': 'Curiosity',
    'images/Spacecrafts/davinci+.png': "Davinci+",
    'images/Spacecrafts/dragonfly.jpeg': 'Dragonfly',
    'images/Spacecrafts/europaclipper.jpeg': 'Europa Clipper',
    'images/Spacecrafts/galileo.jpeg': 'Galileo',
    'images/Spacecrafts/ingenuity.jpeg': 'Ingenuity',
    'images/Spacecrafts/jwst.jpeg': 'JWST',
    'images/Spacecrafts/kepler.jpeg': 'Kepler',
    'images/Spacecrafts/maven.jpeg': 'MAVEN',
    'images/Spacecrafts/mro.jpeg': "MRO",
    'images/Spacecrafts/osirisrex.png': 'OSIRIS-REx',
    'images/Spacecrafts/perserverance.jpeg': 'Perseverance',
    'images/Spacecrafts/philae.jpeg': 'Philae',
    'images/Spacecrafts/phoenix.jpeg': 'Phoenix',
    'images/Spacecrafts/rosetta.jpeg': 'Rosetta',
    'images/Spacecrafts/spirit.jpeg': 'Spirit',
    'images/Spacecrafts/spitzer.jpeg': 'Spitzer',
    'images/Spacecrafts/tess.jpeg': 'TESS',
    'images/Spacecrafts/veritas.jpeg': 'VERITAS',
}

surface_images = {
    "images/MarsLandforms/acidaliaplanitia-1.jpeg": "Acidalia Planitia",
    "images/MarsLandforms/acidaliaplanitia-2.jpeg": "Acidalia Planitia",
    "images/MarsLandforms/albamons.jpeg": "Albamos",
    "images/MarsLandforms/amazonisplanitia.jpeg": "Amazonis Planitia",
    'images/MarsLandforms/aoniaterra.jpeg': 'Aonia Terra',
    "images/MarsLandforms/arabiaterra.jpeg": 'Arabia Terra',
    'images/MarsLandforms/arcadiaplanitia.jpeg': 'Acadia Planitia',
    'images/MarsLandforms/argyreplanitia.jpeg': 'Agyre Planitia',
    'images/MarsLandforms/chryseplanitia.jpeg': 'Chryse Planitia',
    'images/MarsLandforms/cydoniamensae.jpeg': 'Cydonia Mensae',
    'images/MarsLandforms/daedaliaplanum.jpeg': 'Daedalia Planum',
    'images/MarsLandforms/elysium.jpeg': 'Elysium',
    'images/MarsLandforms/hellasplanitia.jpeg': 'Hellas Planitia',
    'images/MarsLandforms/hesperiaplanum.jpeg': "Hesperia Planum",
    'images/MarsLandforms/isidisplanitia.jpeg': "Isidis Planitia",
    'images/MarsLandforms/lunaeplanum.jpeg': 'Lunae Planum',
    'images/MarsLandforms/margaritifierterra.png': 'Margaritifier Terra',
    'images/MarsLandforms/meridianiplanum.jpeg': 'Meridian Planum',
    'images/MarsLandforms/noachisterra.jpeg': 'Noachis Terra',
    'images/MarsLandforms/olympusmons.jpeg': 'Olympus Mons',
    'images/MarsLandforms/prometheiterra.jpeg': 'Promethei Terra',
    'images/MarsLandforms/solisplanum.jpeg': 'Solis Planum',
    'images/MarsLandforms/syrtismajor.jpeg': 'Syrtis Major',
    'images/MarsLandforms/tempeterra.jpeg': 'Tempe Terra',
    'images/MarsLandforms/terracimmeria.jpeg': 'Terra Cimmeria',
    'images/MarsLandforms/terrasabaea.jpeg': 'Terra Asabaea',
    'images/MarsLandforms/terrasirenum.jpg': 'Terra Sirenum',
    'images/MarsLandforms/tharsis.png': 'Tharsis',
    'images/MarsLandforms/thaumasia.jpeg': 'Thaumasia',
    'images/MarsLandforms/tyrrhenaterra.jpeg': 'Tyrrhena Terra',
    'images/MarsLandforms/utopiaplanitia.jpeg': 'Utopia Planitia',
    'images/MarsLandforms/vallesmarineris.jpeg': "Valles Marineris",
    'images/MarsLandforms/xanatheterra.jpeg': "Xanathe Terra",

    'images/MarsFeatures/ancientrivers.jpg': 'Ancient Rivers',
    'images/MarsFeatures/boulders.jpg': 'Boulders',
    'images/MarsFeatures/brainterrain.jpg': 'Brain Terrain',
    'images/MarsFeatures/chaosterrain.jpg': 'Chaos Terrain',
    'images/MarsFeatures/concentriccraterfill.jpg': 'Concentric Crater Fill',
    'images/MarsFeatures/defrosting.jpg': 'defrosting (bruh what)',
    'images/MarsFeatures/delta.jpg': 'Delta',
    'images/MarsFeatures/dune.jpg': 'Dune',
    'images/MarsFeatures/dustdeviltracks.jpg': 'Dust Devil Tracks',
    'images/MarsFeatures/fractureformingboulders.jpg': 'Fracture Forming Boulders',
    'images/MarsFeatures/frettedterrain.jpg': 'Fretted Terrain',
    'images/MarsFeatures/glacier.jpg': 'Glacier',
    'images/MarsFeatures/gullies.jpg': 'Gullies',
    'images/MarsFeatures/gulliesondune.jpg': 'Gullies on Dune',
    'images/MarsFeatures/halocraters.jpg': 'Halo Craters',
    'images/MarsFeatures/icecap.jpg': 'Ice Cap',
    'images/MarsFeatures/icecaplayers.jpg': 'Ice Cap Layers',
    'images/MarsFeatures/latitudedependentmantle.jpg': 'Latitude Dependent Mantle',
    'images/MarsFeatures/lavaflows.jpg': 'Lava Flows',
    'images/MarsFeatures/layeredterrain.jpeg': 'Layered Terrain',
    'images/MarsFeatures/linearridgenetworks.jpg': 'Linear Ridge Networks',
    'images/MarsFeatures/medussaeformation.jpg': "Medussae Formation",
    'images/MarsFeatures/mesas.jpg': 'Mesas',
    'images/MarsFeatures/mudvolcanoes.jpg': 'Mud Volcanoes',
    'images/MarsFeatures/noectislabyrinthus.png': 'Noctis Labyrinths',
    'images/MarsFeatures/pedestalcrater.jpg': 'Pedestal Crater',
    'images/MarsFeatures/polygon.jpg': 'Polygonal Patterned Ground',
    'images/MarsFeatures/reccurentslopelineae.jpg': 'Recurrent Slope Lineae',
    'images/MarsFeatures/ringmoldcraters.jpg': 'Ring Mold Crater',
    'images/MarsFeatures/rootlesscones.jpg': 'Rootless Cones',
    'images/MarsFeatures/scapolledtopography.jpg': 'Scapolled Topography',
    'images/MarsFeatures/slopestreaks.jpeg': 'Slope Streaks',
    'images/MarsFeatures/streamlinedshapes.jpg': 'Stream Lined Shapes',
    'images/MarsFeatures/upperplainsunit.jpg': 'Upper Plains Unit',
    'images/MarsFeatures/volcanoesunderice.jpg': 'Volcanoes Under Ice',
    'images/MarsFeatures/yardangs.jpg': 'Yardangs',

    'images/Maat_Mons_on_Venus.jpg': 'Maat Mons',
    'images/cycloidsoneuropa.jpg': 'Cycloids on Europa',
    'images/labyrinthsontitan.jpg': 'Labyrinths on Titan',
    'images/tigerstripesonenceladus.jpg': 'Tiger Stripes on Enceladus',
}


def compare(text1, text2):
    return util.pytorch_cos_sim(model.encode(text1, convert_to_tensor=True),
                                model.encode(text2, convert_to_tensor=True)).item()


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    with open("leaderboard.json", "r") as f:
        leaderboard = json.load(f)

    if message.author == client.user:
        return

    # Hello Message
    if message.content.startswith('/hello'):
        await message.channel.send('mogo mogo chinese food')

    # Leaderboard
    # First item in tuple is total questions answered and second item in tuple is the answer streak
    if message.content.startswith("/leaderboard"):
        # Did .strip("@") because om doesn't like pings during school
        leaderboard_total_answers = dict(sorted(leaderboard.items(), key=lambda t: t[1][0], reverse=True))
        leaderboard_answer_streak = dict(sorted(leaderboard.items(), key=lambda t: t[1][1], reverse=True))

        leaderboard_total_answer_str = "*Total Answers* \n" + "".join(f"{user}: {leaderboard[user][0]} \n" for user in leaderboard_total_answers.keys())
        leaderboard_answer_streak_str = "*Answer Steaks:* \n" + "".join(f"{user}: {leaderboard[user][1]} \n" for user in leaderboard_answer_streak.keys())

        await message.channel.send("***Leaderboard:*** \n \n" + leaderboard_total_answer_str + "\n" + leaderboard_answer_streak_str)

    # Nick Message
    if message.content.startswith("/athiwat"):
        await message.channel.send('Athiwat is Amazing! Athiwat is MEGAcracked!')

    # Image Id
    if message.content.startswith('/imageid'):
        random_float = random.uniform(0, 1)

        # Because we will not to know that much Surface Id we only choose those images 25% of the time
        if 0 <= random_float < 0.4:
            imageQuestion = random.choice(list(extrasolar_images.keys()))
            imageAnswer = extrasolar_images[imageQuestion]
        if 0.4 <= random_float < 0.75:
            imageQuestion = random.choice(list(spacecrafts_images.keys()))
            imageAnswer = spacecrafts_images[imageQuestion]
        if 0.75 <= random_float <= 1:
            imageQuestion = random.choice(list(surface_images.keys()))
            imageAnswer = surface_images[imageQuestion]

        await message.channel.send(file=discord.File(imageQuestion))

        answer_message = await client.wait_for("message", check=lambda m: m.author == message.author)

        if answer_message.content.lower() == imageAnswer.lower():
            await message.channel.send(f"{message.author.mention} correct u cracked bro correct answer: {imageAnswer}")

            # Leaderboard
            if message.author.mention in leaderboard:
                leaderboard[message.author.mention][0] += 1
                leaderboard[message.author.mention][1] += 1
            else:
                leaderboard[message.author.mention] = [1, 1]
        else:
            await message.channel.send(f'{message.author.mention} incorrect u suck. correct answer: {imageAnswer}')

            # Leaderboard
            if message.author.mention in leaderboard:
                leaderboard[message.author.mention][1] = 0
            else:
                leaderboard[message.author.mention] = [0, 0]

        # Dumps leaderboard
        with open('leaderboard.json', 'w') as f:
            json.dump(leaderboard, f)

    # Anki Questions
    if message.content.startswith('/anki'):
        question = random.choice(list(questions.keys()))

        await message.channel.send(question)

        answer_message = await client.wait_for("message", check=lambda m: m.author == message.author)

        answer_message = answer_message.content

        answer = questions[question]

        accuracy = compare(str(answer_message).lower(), str(answer).lower())

        if str(answer_message).lower() == str(answer).lower():
            await message.channel.send(
                f"{message.author.mention} \n `Correct!` ur cracked \n `Accuracy:` {round(accuracy * 100, 2)}%. Bruh \n `Correct Answer:` {answer}")

            # Leaderboard
            if message.author.mention in leaderboard:
                leaderboard[message.author.mention][0] += 1
                leaderboard[message.author.mention][1] += 1
            else:
                leaderboard[message.author.mention] = [1, 1]

        else:
            if accuracy > 0.58:
                await message.channel.send(
                    f"{message.author.mention} \n `Correct!`` ur cracked \n ``Accuracy:`` {round(accuracy * 100, 2)}%. Bruh \n ``Correct Answer:`` {answer}")

                # Leaderboard
                if message.author.mention in leaderboard:
                    leaderboard[message.author.mention][0] += 1
                    leaderboard[message.author.mention][1] += 1
                else:
                    leaderboard[message.author.mention] = [1, 1]

            else:
                await message.channel.send(
                    f"{message.author.mention} \n ``Incorrect`` u suck \n ``Accuracy:`` {round(accuracy * 100, 2)}%. Bruh \n `Correct Answer:` {answer}")

                # Leaderboard
                if message.author.mention in leaderboard:
                    leaderboard[message.author.mention][1] = 0
                else:
                    leaderboard[message.author.mention] = [0, 0]

        # Dumps leaderboard
        with open('leaderboard.json', 'w') as f:
            json.dump(leaderboard, f)

        flag_question = await client.wait_for("message", check=lambda m: m.author == message.author)

        if flag_question.content.startswith("/flag"):
            # Reason why the question was flagged
            flag_reason = flag_question.content.split("/flag")[1].strip(" ")
            with open("flag.txt", mode="a", encoding="utf-8") as f:
                f.write(f"Question: {question} --- Reason for flag: {flag_reason} \n")

            await message.channel.send(
                "Thanks for flagging this question (make sure u have a good reason doe). Ayo nick moment")


client.run(token)
