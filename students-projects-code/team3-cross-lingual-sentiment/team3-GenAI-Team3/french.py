import pandas as pd

ground_truth_fr = [
    {
        "topic": "reunion_fed_us_mai_2025",
        "generated": "",
        "references": [
            "Le 5 mai, Boursorama Marchés affirme que le consensus attribue 97 % de probabilité à un statu quo de la Fed, laissant le corridor à 4,25‑4,50 %.",
            "La Tribune (3 mai) estime que, malgré les pressions de Donald Trump, la Fed ne devrait pas modifier ses taux lors de la réunion des 6‑7 mai.",
            "Dans sa chronique Doze d’économie (BFM Business, 5 mai), Nicolas Doze souligne que le reflux du pétrole à 60 $ reflète l’attentisme avant le FOMC.",
            "Zonebourse (5 mai) note que Wall Street marque une pause et que le dollar reste sous pression dans l’attente des annonces de la Fed.",
            "L’Agefi (5 mai) juge que la banque centrale patientera encore plusieurs mois avant d’envisager une première baisse de taux."
        ],
    },
    {
        "topic": "trump_troisieme_mandat_tarifs",
        "generated": "",
        "references": [
            "France 24 (5 mai) rapporte la volonté de Donald Trump d’imposer un droit de douane de 100 % sur tous les films produits à l’étranger afin de « sauver Hollywood ».",
            "Un second article de France 24 décrit la consternation des studios face à cette mesure jugée anxiogène pour l’industrie mondiale.",
            "Le Figaro résume, dans ses « 5 infos à connaître » du 5 mai, qu’il s’agit d’une nouvelle escalade dans la politique commerciale du président américain.",
            "Yahoo Actualités‑AFP (5 mai) souligne que le périmètre exact (streaming, coproductions) reste flou et inquiète fortement le secteur.",
            "Reuters explique que le département du Commerce devra préciser la base taxable et que plusieurs capitales alliées préparent déjà des représailles."
        ],
    },
    {
        "topic": "inondations_bahia_bresil_mai_2025",
        "generated": "",
        "references": [
            "Le 1ᵉʳ mai, Catnat.net émet une alerte évoquant des pluies « excessives à extrêmes » attendues sur le littoral de Bahia jusqu’au 10 mai.",
            "La veille‑catastrophes précise que l’épisode pourrait apporter entre 150 et 400 mm de précipitations en quelques jours.",
            "Le ministère français des Affaires étrangères rappelle que la saison humide au Brésil provoque régulièrement crues et glissements de terrain.",
            "Météo‑Suisse met en contexte les cumuls records début mai, comparant 300 mm en 72 h à trois mois de pluie pour la région concernée.",
            "Catnat.net ajoute que le cyclone subtropical au large renforce l’instabilité et le risque d’inondations urbaines à Salvador."
        ],
    },
    {
        "topic": "plan_israel_gaza",
        "generated": "",
        "references": [
            "Le Monde (5 mai) titre sur l’approbation d’un plan prévoyant la « conquête » complète de la bande de Gaza avec déplacement interne des habitants.",
            "France 24‑AFP rapporte que l’armée envisage de déplacer « la plupart » des Gazaouis vers le sud et de contrôler les couloirs d’aide.",
            "Une analyse vidéo de France 24 détaille les objectifs militaires et le calendrier supposé de l’opération.",
            "BFM TV relaie la confirmation officielle du feu vert du cabinet de sécurité israélien à cette extension offensive.",
            "The Guardian (version française) souligne les mises en garde de l’ONU contre un possible crime de guerre lié au déplacement forcé."
        ],
    },
    {
        "topic": "licenciements_tech_2025",
        "generated": "",
        "references": [
            "ITRnews dresse début mai un total de près de 53 000 suppressions de postes dans la tech mondiale depuis janvier.",
            "Un reportage TF1 Info (2 mai) avertit que des centaines de milliers d’emplois industriels et numériques pourraient être menacés en France.",
            "L’Informaticien annonce le 2 mai que STMicroelectronics va supprimer 1 000 emplois en France dans un plan global de 2 800 coupes.",
            "Mobilicités revient le 2 mai sur la perte de 1 000 emplois chez un grand fabricant européen de semi‑conducteurs.",
            "La Tribune révèle le 30 avril qu’Intel prépare un licenciement de 20 % de ses effectifs, confirmant la poursuite des coupes chez les géants du secteur."
        ],
    },
]

gt_df_fr = pd.DataFrame(ground_truth_fr)

print(gt_df_fr.head())
