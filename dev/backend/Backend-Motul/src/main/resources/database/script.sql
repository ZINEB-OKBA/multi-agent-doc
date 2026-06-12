-- 27/10/2025--
-- Modules--
INSERT INTO public.module (code, libelle, ordre) VALUES (1, 'ADMINISTRATION', 1);
INSERT INTO public.module (code, libelle, ordre) VALUES (2, 'ASSISTANT_CONVERSATIONNEL', 2);
INSERT INTO public.module (code, libelle, ordre) VALUES (3, 'DATABASE', 3);

--Profiles--
INSERT INTO public.profiles (id, created_by, created_on, deleted, updated_by, updated_on, name) VALUES (1, null, null, false, null, null, 'Administrateur');
INSERT INTO public.profiles (id, created_by, created_on, deleted, updated_by, updated_on, name) VALUES (2, null, null, false, null, null, 'Utilisateur');

--Roles--
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_AJOUTER_UTILISATEUR', 'Ajouter un utilisateur', 'Ajouter ', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_SUPPRIMER_UTILISATEUR', 'Supprimer un utilisateur', 'Supprimer', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_MODIFIER_UTILISATEUR', 'Modifier un utilisateur', 'Modifier ', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_AFFICHER_UTILISATEUR', 'Afficher un utilisateur', 'Afficher', 1);

INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_AJOUTER_HABILITATION', 'Ajouter une habilitation', 'Ajouter habilitation', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_MODIFIER_HABILITATION', 'Modifier une habilitation', 'Modifier', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_AFFICHER_HABILITATION', 'Afficher une habilitation', 'Afficher', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_SUPPRIMER_HABILITATION', 'Supprimer une habilitation ', 'Supprimer', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_GESTION_UTILISATEUR', 'Gestion des utilisateur ', 'Gestion', 1);
INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_ASSISTANT_CONVERSATIONNEL', 'Afficher Assistant conversationnel ', 'Afficher', 2);

INSERT INTO public.role (identifiant, description, libelle, module) VALUES ('ROLE_AFFICHER_DATABASE', 'Afficher la base de données', 'Afficher', 3);


--Profile Roles--
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_AJOUTER_UTILISATEUR');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_SUPPRIMER_UTILISATEUR');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_MODIFIER_UTILISATEUR');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_AFFICHER_UTILISATEUR');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_AJOUTER_HABILITATION');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_MODIFIER_HABILITATION');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_AFFICHER_HABILITATION');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_SUPPRIMER_HABILITATION');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_GESTION_UTILISATEUR');
INSERT INTO public.profil_roles (profil, roles) VALUES (1, 'ROLE_ASSISTANT_CONVERSATIONNEL');
INSERT INTO public.profil_roles (profil, roles) VALUES (2, 'ROLE_ASSISTANT_CONVERSATIONNEL');