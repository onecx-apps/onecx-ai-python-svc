-- --keycloak
CREATE USER keycloak WITH ENCRYPTED PASSWORD 'keycloak';
CREATE DATABASE keycloak with owner keycloak;
GRANT ALL PRIVILEGES ON DATABASE keycloak TO keycloak;
-- --tkit-portal-server
CREATE USER "tkit-portal-server" WITH ENCRYPTED PASSWORD 'tkit-portal-server';
CREATE DATABASE "tkit-portal-server" with owner "tkit-portal-server";
GRANT ALL PRIVILEGES ON DATABASE "tkit-portal-server" TO "tkit-portal-server";
-- --apm
CREATE USER apm WITH ENCRYPTED PASSWORD 'apm';
CREATE DATABASE apm with owner apm;
GRANT ALL PRIVILEGES ON DATABASE apm TO apm;
-- --ahm
CREATE USER ahm WITH ENCRYPTED PASSWORD 'ahm';
CREATE DATABASE ahm with owner ahm;
GRANT ALL PRIVILEGES ON DATABASE ahm TO ahm;
-- --data mgmt
CREATE USER datamgmt WITH ENCRYPTED PASSWORD 'datamgmt';
CREATE DATABASE datamgmt with owner datamgmt;
GRANT ALL PRIVILEGES ON DATABASE datamgmt TO datamgmt;
-- --event-management
CREATE USER "event-management" WITH ENCRYPTED PASSWORD 'event-management';
CREATE DATABASE "event-management" with owner "event-management";
GRANT ALL PRIVILEGES ON DATABASE "event-management" TO "event-management";
-- --location-management
CREATE USER "location-mgmt" WITH ENCRYPTED PASSWORD 'location-mgmt';
CREATE DATABASE "location-mgmt" with owner "location-mgmt";
GRANT ALL PRIVILEGES ON DATABASE "location-mgmt" TO "location-mgmt";
-- --mailing-service
CREATE USER "mailing_service" WITH ENCRYPTED PASSWORD 'mailing_service';
CREATE DATABASE "mailing_service" with owner "mailing_service";
GRANT ALL PRIVILEGES ON DATABASE "mailing_service" TO "mailing_service";
-- --event-management-touchpoint
CREATE USER "event-management-touchpoint" WITH ENCRYPTED PASSWORD 'event-management-touchpoint';
CREATE DATABASE "event-management-touchpoint" with owner "event-management-touchpoint";
GRANT ALL PRIVILEGES ON DATABASE "event-management-touchpoint" TO "event-management-touchpoint";
-- --document-management
CREATE USER "documentmanagement" WITH ENCRYPTED PASSWORD 'documentmanagement';
CREATE DATABASE "documentmanagement" with owner "documentmanagement";
GRANT ALL PRIVILEGES ON DATABASE "documentmanagement" TO "documentmanagement";
-- --wifi-access-mgmt
CREATE USER "wifi-access-management" WITH ENCRYPTED PASSWORD 'wifi-access-management';
CREATE DATABASE "wifi-access-management" with owner "wifi-access-management";
GRANT ALL PRIVILEGES ON DATABASE "wifi-access-management" TO "wifi-access-management";
-- --organization-management
CREATE USER "organization_management" WITH ENCRYPTED PASSWORD 'organization_management';
CREATE DATABASE "organization_management" with owner "organization_management";
GRANT ALL PRIVILEGES ON DATABASE "organization_management" TO "organization_management";
-- --individual-management
CREATE USER "individual_management" WITH ENCRYPTED PASSWORD 'individual_management';
CREATE DATABASE "individual_management" with owner "individual_management";
GRANT ALL PRIVILEGES ON DATABASE "individual_management" TO "individual_management";
-- --token-service
CREATE USER "token_service" WITH ENCRYPTED PASSWORD 'token_service';
CREATE DATABASE "token_service" with owner "token_service";
GRANT ALL PRIVILEGES ON DATABASE "token_service" TO "token_service";
-- --equipment-mgmt
CREATE USER "equipment-mgmt" WITH ENCRYPTED PASSWORD 'equipment-mgmt';
CREATE DATABASE "equipment-mgmt" with owner "equipment-mgmt";
GRANT ALL PRIVILEGES ON DATABASE "equipment-mgmt" TO "equipment-mgmt";
