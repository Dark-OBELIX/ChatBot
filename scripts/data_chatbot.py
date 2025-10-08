# scripts/data_chatbot.py

data = [
    # --- 1. Salutations ---
    {
        "question": [
            "Bonjour", "Salut", "Coucou", "Hey", "Hello", "Yo", "Salut Mobilia", "Bonjour le bot",
            "Bonsoir", "Salut à toi", "Bonjour MobiliaBot", "Salut le robot", "Hey salut",
            "Bonne journée", "Hello Mobilia", "Salut tout le monde", "Coucou Mobilia",
            "Bonsoir à vous", "Rebonjour", "Salut l’assistant Mobilia", "Salut le service client"
        ],
        "answer": "Bonjour 👋 Bienvenue chez Mobilia ! Comment puis-je vous aider aujourd’hui ?",
        "action_index": 0
    },

    # --- 2. Présentation / Identité ---
    {
        "question": [
            "Qui es-tu ?", "C’est quoi Mobilia ?", "Tu fais quoi ?", "Que peux-tu faire ?",
            "Explique-moi ton rôle", "T’es qui toi ?", "Quel est ton but ?", "C’est quoi ton job ?",
            "Tu travailles pour qui ?", "Tu es un vrai humain ?", "Tu appartiens à Mobilia ?",
            "Tu fais partie de l’équipe ?", "Es-tu un conseiller ?", "Quel est ton rôle exact ?",
            "Parle-moi de toi", "Qui t’a créé ?", "Qui t’a développé ?", "Que fais-tu concrètement ?",
            "Tu représentes Mobilia ?", "À quoi sers-tu ?"
        ],
        "answer": "Je suis MobiliaBot 🤖, votre assistant intelligent pour tout savoir sur nos meubles, livraisons et services.",
        "action_index": 0
    },

    # --- 3. Produits disponibles ---
    {
        "question": [
            "Quels produits vendez-vous ?", "Tu proposes quoi ?", "Quels types de meubles avez-vous ?",
            "Vous avez des canapés ?", "Tu as des chaises ?", "Vous vendez des lits ?", "Tu as des tables ?",
            "Tu proposes des bureaux ?", "Vous avez des armoires ?", "Je cherche un meuble TV",
            "Avez-vous du mobilier de jardin ?", "Proposez-vous des luminaires ?", "Vous avez des lampes ?",
            "Vous vendez des meubles en bois ?", "Avez-vous des fauteuils ?", "Je cherche une table basse",
            "Tu as des étagères ?", "Quel type de mobilier proposez-vous ?", "Tu peux me montrer vos produits ?",
            "Je veux voir le catalogue"
        ],
        "answer": "Nous proposons un large choix : tables, chaises, canapés, lits, bureaux, meubles TV, rangements et luminaires 💡.",
        "action_index": 0
    },

    # --- 4. Commandes / Achat ---
    {
        "question": [
            "Comment passer commande ?", "Je veux acheter un meuble", "Comment commander ?",
            "Je veux passer une commande", "Tu peux m’aider à commander ?", "Je veux acheter",
            "Comment faire un achat ?", "Comment valider mon panier ?", "J’ai ajouté au panier et après ?",
            "Comment payer ?", "Je veux finaliser ma commande", "Je ne trouve pas le bouton de paiement",
            "Comment confirmer ma commande ?", "Je souhaite passer une commande", "Je veux acheter un canapé",
            "Aide-moi à acheter une table", "Je veux un bureau en bois clair", "Comment réserver un produit ?",
            "Puis-je commander par téléphone ?", "Où dois-je cliquer pour acheter ?"
        ],
        "answer": "C’est simple 🛒 : ajoutez vos produits à votre panier, puis validez la commande sur mobilia.fr. Je peux aussi vous guider !",
        "action_index": 0
    },

    # --- 5. Suivi de commande ---
    {
        "question": [
            "Où en est ma commande ?", "Je veux suivre ma commande", "Où est mon colis ?",
            "Quand vais-je recevoir ma commande ?", "Peux-tu me dire quand j’aurai ma table ?",
            "Suivi de commande", "Statut de ma livraison", "C’est livré quand ?", "Où est mon meuble ?",
            "Je n’ai toujours rien reçu", "Mon colis n’est pas arrivé", "Combien de temps reste-t-il ?",
            "Le transporteur ne m’a pas contacté", "Comment suivre la livraison ?", "Je veux voir le suivi",
            "J’attends un canapé", "Ma commande est bloquée", "Où se trouve ma livraison ?",
            "Ma commande est partie ?", "Quand la livraison aura lieu ?"
        ],
        "answer": "Pour suivre votre commande, entrez votre numéro sur la page *Suivi de commande* de mobilia.fr 📦.",
        "action_index": 0
    },

    # --- 6. Livraison ---
    {
        "question": [
            "Quels sont vos délais de livraison ?", "Quand vais-je recevoir mon meuble ?",
            "Vous livrez en combien de temps ?", "Combien de jours pour la livraison ?",
            "Vous livrez à domicile ?", "Livraison express disponible ?", "Livrez-vous en Belgique ?",
            "Est-ce que vous livrez en Suisse ?", "Combien coûte la livraison ?", "La livraison est gratuite ?",
            "Quels sont vos transporteurs ?", "Puis-je choisir la date de livraison ?",
            "Livrez-vous le week-end ?", "Je veux changer l’adresse de livraison",
            "Livrez-vous à l’étranger ?", "Puis-je être livré sur mon lieu de travail ?",
            "Avez-vous une option premium ?", "Quel est le délai moyen ?", "Combien de temps pour un canapé ?",
            "Est-ce que vous faites le montage à la livraison ?"
        ],
        "answer": "Nos livraisons prennent entre 3 et 7 jours ouvrés 🚚. Oui, nous livrons à domicile en France, Belgique et Suisse.",
        "action_index": 0
    },

    # --- 7. Retours / SAV ---
    {
        "question": [
            "Je veux retourner un produit", "Comment faire un retour ?", "Mon meuble est cassé",
            "Je veux un échange", "J’ai reçu un produit défectueux", "Comment fonctionne le SAV ?",
            "J’ai un problème avec ma commande", "Puis-je être remboursé ?", "Comment contacter le service client ?",
            "Le meuble ne me plaît pas", "Je veux changer de couleur", "Le colis est abîmé",
            "Je n’ai pas reçu les bonnes pièces", "Ma table est rayée", "Le montage est impossible",
            "Il manque une vis", "J’ai reçu la mauvaise couleur", "Je souhaite un remboursement complet",
            "Le livreur a abîmé mon colis", "Puis-je renvoyer l’article ?"
        ],
        "answer": "Pas d’inquiétude ! Contactez notre service après-vente via le formulaire *Assistance Mobilia*, nous prendrons tout en charge 🔧.",
        "action_index": 0
    },

    # --- 8. Conseils déco ---
    {
        "question": [
            "Peux-tu me conseiller ?", "J’hésite entre deux modèles", "Quel style choisir ?",
            "Tu peux m’aider pour ma déco ?", "Quelle couleur irait dans un salon clair ?",
            "Quel canapé pour un petit salon ?", "Quel bureau pour télétravailler ?",
            "Quel style est à la mode ?", "Aide-moi à choisir un meuble", "J’ai un salon beige, que mettre ?",
            "Quelle table pour un salon moderne ?", "Quel luminaire pour un bureau ?",
            "Quelle matière est la plus solide ?", "Tu me conseilles quel style scandinave ?",
            "J’aime le bois clair, tu proposes quoi ?", "Quel canapé pour un grand salon ?",
            "Je cherche un meuble minimaliste", "Je veux une ambiance industrielle",
            "Tu as des idées déco ?", "Tu peux m’aider à harmoniser les couleurs ?"
        ],
        "answer": "Avec plaisir 🎨 ! Pour un salon clair, privilégiez les tons beiges et le bois clair. Pour un petit espace, optez pour du mobilier modulable.",
        "action_index": 0
    },

    # --- 9. Prix et promotions ---
    {
        "question": [
            "Quels sont vos prix ?", "C’est cher ?", "Vous avez des promotions ?",
            "Y a-t-il des soldes ?", "Faites-vous des réductions ?", "Je veux une remise",
            "Quel est le prix d’un canapé ?", "Combien coûte une table en bois ?",
            "Vous avez des codes promo ?", "Quand sont les soldes ?", "Y a-t-il des ventes privées ?",
            "Puis-je avoir un bon de réduction ?", "Vous faites le Black Friday ?",
            "Est-ce qu’il y a des offres spéciales ?", "Vos prix incluent la TVA ?", "Le montage est compris ?",
            "Les frais de livraison sont inclus ?", "Vous avez des packs ?",
            "C’est moins cher en magasin ?", "Comment profiter d’une promo ?"
        ],
        "answer": "Nos prix varient selon la gamme 💶. Consultez mobilia.fr pour voir les promos et nos soldes en cours !",
        "action_index": 0
    },

    # --- 10. Paiement ---
    {
        "question": [
            "Quels moyens de paiement acceptez-vous ?", "Puis-je payer en plusieurs fois ?",
            "Acceptez-vous PayPal ?", "Je veux payer en 3 fois", "Puis-je payer à la livraison ?",
            "Vous prenez les cartes Visa ?", "Le paiement est sécurisé ?", "Je peux payer en chèque ?",
            "Le paiement se fait comment ?", "Je peux payer par virement ?", "Vous acceptez Apple Pay ?",
            "Puis-je utiliser Klarna ?", "Puis-je avoir une facture ?", "Le paiement a échoué",
            "Puis-je utiliser ma carte cadeau ?", "Acceptez-vous Mastercard ?",
            "Je veux payer plus tard", "Vous acceptez le paiement en crypto ?", "Le paiement est bien validé ?",
            "Je veux changer de mode de paiement"
        ],
        "answer": "Nous acceptons carte bancaire, PayPal et paiement en 3x sans frais 💳.",
        "action_index": 0
    },

    # --- 11. Contact / Magasins ---
    {
        "question": [
            "Comment vous contacter ?", "Avez-vous un numéro de téléphone ?", "Je veux parler à un humain",
            "Quels sont vos horaires ?", "Vous êtes ouverts le dimanche ?", "Je peux passer au magasin ?",
            "Où se trouve votre boutique ?", "Adresse Mobilia ?", "Avez-vous un showroom ?",
            "Où sont vos magasins ?", "Je veux l’adresse du magasin de Bordeaux",
            "Je veux venir sur place", "Avez-vous un magasin à Paris ?", "Pouvez-vous m’appeler ?",
            "Le service client est ouvert ?", "Je veux écrire un mail", "Comment parler à un conseiller ?",
            "Je veux prendre rendez-vous", "Pouvez-vous me rappeler ?", "Avez-vous un tchat humain ?"
        ],
        "answer": "Notre service client est joignable du lundi au samedi de 9h à 19h au 09 72 00 00 00 ☎️ ou en boutique à Bordeaux et Paris.",
        "action_index": 0
    },

    # --- 12. Remerciements ---
    {
        "question": [
            "Merci", "Merci beaucoup", "Je te remercie", "C’est gentil", "Merci pour ton aide",
            "Super merci", "Merci infiniment", "Merci MobiliaBot", "Top merci", "T’es super",
            "Merci du renseignement", "Merci d’avance", "Merci pour tes conseils", "Je te remercie encore",
            "C’est top", "Trop bien", "Parfait merci", "Merci à toi", "Merci pour tout", "Tu m’as bien aidé"
        ],
        "answer": "Avec grand plaisir 😊. N’hésitez pas à revenir si vous avez d’autres questions !",
        "action_index": 0
    },

    # --- 13. Clôture / Fin de discussion ---
    {
        "question": [
            "Au revoir", "À bientôt", "Bonne soirée", "Bonne nuit", "Ciao", "Je te laisse",
            "Je pars", "Merci, c’est tout", "Je n’ai plus de question", "Bonne journée à toi",
            "À plus tard", "À la prochaine", "C’est fini pour moi", "Je quitte la conversation",
            "À demain", "On se revoit bientôt", "Bye bye", "Je te dis au revoir", "Fin de chat", "Je clos ici"
        ],
        "answer": "À très bientôt chez Mobilia 🪑 !",
        "action_index": 1
    }
]
