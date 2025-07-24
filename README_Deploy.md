# 🚀 Guía de Despliegue del Dashboard PVStand

## 🌐 Opción 1: Streamlit Cloud (Recomendada)

### Pasos para desplegar en Streamlit Cloud:

1. **Crear cuenta en Streamlit Cloud:**
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Regístrate con tu cuenta de GitHub

2. **Preparar el repositorio:**
   ```bash
   # Crear un repositorio en GitHub
   git init
   git add .
   git commit -m "Dashboard PVStand IV Curves"
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```

3. **Desplegar en Streamlit Cloud:**
   - Conecta tu repositorio de GitHub
   - Selecciona el archivo: `dashboard_pvstand_curves.py`
   - Configura las variables de entorno si es necesario
   - ¡Listo! Tu dashboard tendrá una URL pública

### Ventajas:
- ✅ Gratuito
- ✅ URL pública automática
- ✅ Actualizaciones automáticas desde GitHub
- ✅ Sin configuración de servidor

---

## 🌐 Opción 2: ngrok (Para pruebas rápidas)

### Instalar ngrok:
```bash
# Descargar ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz

# Registrar en ngrok.com y obtener authtoken
./ngrok authtoken TU_TOKEN
```

### Exponer el dashboard:
```bash
# Ejecutar el dashboard
cd /home/nicole/SR/SOILING
source .venv/bin/activate
streamlit run dashboard_pvstand_curves.py --server.port 8501

# En otra terminal, exponer con ngrok
./ngrok http 8501
```

### Ventajas:
- ✅ Rápido de configurar
- ✅ URL temporal pública
- ✅ Ideal para demostraciones

### Desventajas:
- ❌ URL cambia cada vez que reinicias
- ❌ Limitaciones de uso gratuito

---

## 🌐 Opción 3: VPS/Cloud Server

### Configurar servidor:
```bash
# En tu servidor VPS
sudo apt update
sudo apt install python3 python3-pip nginx

# Clonar el proyecto
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO/SOILING

# Instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configurar systemd service
sudo nano /etc/systemd/system/pvstand-dashboard.service
```

### Archivo de servicio systemd:
```ini
[Unit]
Description=PVStand Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/TU_REPO/SOILING
Environment=PATH=/home/ubuntu/TU_REPO/SOILING/.venv/bin
ExecStart=/home/ubuntu/TU_REPO/SOILING/.venv/bin/streamlit run dashboard_pvstand_curves.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

### Configurar Nginx:
```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🔧 Configuración de Datos

### Para acceso remoto a ClickHouse:
```python
# En dashboard_pvstand_curves.py, modificar la configuración:
CLICKHOUSE_CONFIG = {
    'host': "TU_IP_PUBLICA",  # IP pública del servidor ClickHouse
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}
```

### Variables de entorno recomendadas:
```bash
# Crear archivo .env
CLICKHOUSE_HOST=146.83.153.212
CLICKHOUSE_PORT=30091
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=Psda2020
```

---

## 📊 Monitoreo y Mantenimiento

### Logs del dashboard:
```bash
# Ver logs en tiempo real
journalctl -u pvstand-dashboard -f

# Reiniciar servicio
sudo systemctl restart pvstand-dashboard
```

### Backup de datos:
```bash
# Script de backup automático
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp /home/nicole/SR/SOILING/datos/raw_pvstand_curves_data.csv /backup/pvstand_curves_$DATE.csv
```

---

## 🚨 Consideraciones de Seguridad

1. **Firewall:** Configurar solo puertos necesarios
2. **HTTPS:** Usar certificados SSL para producción
3. **Autenticación:** Considerar agregar login si es necesario
4. **Rate Limiting:** Limitar requests por IP
5. **Backup:** Respaldar datos regularmente

---

## 📞 Soporte

Si tienes problemas con el despliegue:
1. Revisa los logs del servicio
2. Verifica la conectividad de red
3. Confirma que las dependencias están instaladas
4. Consulta la documentación de Streamlit Cloud 